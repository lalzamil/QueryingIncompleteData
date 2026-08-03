"""
Ranking query rewriter + executor: pattern-based top-fraction with optional
interval bounds.  Adapted from V1's fast SQL structure:
  - Merged stats CTE (single scan with P_H), theta-based H for speed
  - Certain tuples bypass ranking (score=1, always returned)
  - Uncertain tuples ranked at the TUPLE level (no reverse JOIN)
  - GROUP BY payload at the end for deduplication
  - Interval width computed as a lightweight post-query (full H)
  - Composite index + ANALYZE (V1 style)
"""
import re, time as _time
from collections import defaultdict
from SetQueryRewriterExecuter import (
    QueryExecutor as BaseQueryExecutor,
    QueryRewriter as BaseQueryRewriter,
    _z_critical, _interval_bounds_sql, _delta_relvar_sql,
)

_RES = frozenset({'default','order','group','select','where','from','table',
    'column','index','check','primary','key','user','type','value','name',
    'count','all','having','limit','offset','row','join','using','as','on',
    'with','and','or','not','in','is','null','true','false'})
def _qn(n):
    return f'"{n}"' if n.lower() in _RES else n


class QueryRewriterRanking(BaseQueryRewriter):

    def _q(self, cond, alias, cols):
        out = cond
        for c in sorted(cols, key=len, reverse=True):
            out = re.sub(r'\b' + re.escape(c) + r'\b', f'{alias}.{c}',
                         out, flags=re.IGNORECASE)
        return out

    def groupLevelQueryRewriter(self, fraction=0.5, iv_mode=None,
                                iv_alpha=0.05):
        if not getattr(self, "table", None):
            return self.base_query, 0, 0, [], False, None
        al = self.base_alias
        tbl = (self.table[:-4] if self.table.lower().endswith(".csv")
               else self.table)
        phi_miss = [f for f in self.factors
                    if f["name"] in self.missing_attrs]
        phi_obs = [c for c in self.where_conds
                   if not any(a in c for a in self.missing_attrs)]
        all_cond = {col for f in phi_miss for col in f["conditioning"]}
        if not all_cond:
            return self.base_query, 0, 0, [], False, None

        theta_cond = {col for f in phi_miss if f["condition"] != "TRUE"
                      for col in f["conditioning"]}
        inGrp = [c.strip()
                 for c in getattr(self, "group_by_cols", []) if c.strip()]
        activeGrp = sorted(theta_cond) if theta_cond else []
        if inGrp and set(inGrp).issubset(theta_cond or all_cond):
            activeGrp = inGrp
        forced = list(dict.fromkeys(
            getattr(self, "force_group_cols", []) or []))
        if forced:
            s = set(activeGrp)
            activeGrp = activeGrp + [c for c in forced if c not in s]

        sel_extra = [c for c in self.select_cols
                     if c not in set(activeGrp)]
        pay_cols = self.select_cols
        known = (set(all_cond) | set(self.missing_attrs)
                 | set(activeGrp) | set(pay_cols))
        obs_al = [self._q(c, al, known) for c in phi_obs]
        obs_raw = [f"({c})" for c in phi_obs]

        # 1) Merged-stats CTE (theta-based H for speed)
        agg_parts = ["COUNT(*) AS count_h"]
        for f in phi_miss:
            nm = f["name"]; qnm = _qn(nm)
            agg_parts.append(
                f"AVG(CASE WHEN {qnm} IS NOT NULL "
                f"THEN (CASE WHEN {f['condition']} THEN 1.0 ELSE 0.0 END) "
                f"ELSE NULL END) AS p_{nm}")
            agg_parts.append(f"NULLIF(COUNT({qnm}), 0) AS den_{nm}")

        grp_list = ", ".join(_qn(c) for c in activeGrp)
        where_stats = " AND ".join(obs_raw) if obs_raw else "TRUE"
        sel_gs = (f"{grp_list}, " if grp_list else "") + ", ".join(agg_parts)
        gb_gs = f"\n  GROUP BY {grp_list}" if grp_list else ""
        grouped = (f"gs AS (\n  SELECT {sel_gs}\n"
                   f"  FROM {tbl}\n  WHERE {where_stats}{gb_gs}\n)")
        merged = (f"ms AS (\n  SELECT gs.*,\n"
                  f"    gs.count_h * 1.0 / NULLIF(SUM(gs.count_h) "
                  f"OVER (), 0) AS p_h\n  FROM gs\n)")

        # 2) MV CTE (NOT MATERIALIZED)
        miss_ind = [f"{f['name']}_miss" for f in phi_miss]
        miss_sql_parts = [
            f"(CASE WHEN {al}.{f['name']} IS NULL THEN 1 ELSE 0 END) "
            f"AS {f['name']}_miss" for f in phi_miss]
        msum = (" + ".join(
            f"(CASE WHEN {al}.{f['name']} IS NULL THEN 1 ELSE 0 END)"
            for f in phi_miss) or "0")
        allow_miss = []
        for f in phi_miss:
            nm = f["name"]
            cq = self._q(f["condition"], al,
                         set(f["conditioning"]) | {nm})
            allow_miss.append(f"({al}.{nm} IS NULL OR {cq})")
        mv_where = " AND ".join(allow_miss + obs_al) or "TRUE"
        grp_mv = ", ".join(f"{al}.{c}" for c in activeGrp)
        ext_mv = ", ".join(f"{al}.{c}" for c in sel_extra)
        mv_parts = [x for x in [grp_mv, ext_mv] if x]
        sel_mv = ", ".join(mv_parts)
        pay_mv = ", ".join(f"mv.{c}" for c in pay_cols)
        mv_cte = (f"mv AS NOT MATERIALIZED (\n"
                  f"  SELECT {sel_mv},\n"
                  f"    {', '.join(miss_sql_parts)},\n"
                  f"    ({msum})::int AS miss_sum\n"
                  f"  FROM {tbl} {al}\n  WHERE {mv_where}\n)")

        # 3) Patterns CTE (uncertain only, theta-based H)
        gp_mv = ", ".join(f"mv.{c}" for c in activeGrp)
        mp_mv = ", ".join(f"mv.{i}" for i in miss_ind)
        pat_sel = [x for x in [gp_mv, mp_mv] if x]
        pgb = ", ".join(pat_sel) if pat_sel else mp_mv
        pat_cte = (f"patterns AS NOT MATERIALIZED (\n"
                   f"  SELECT {', '.join(pat_sel)},\n"
                   f"    COUNT(*) AS mv_count\n"
                   f"  FROM mv WHERE mv.miss_sum > 0\n"
                   f"  GROUP BY {pgb}\n)")

        # 4) Pattern probabilities (pp)
        ms_join = (" AND ".join(
            f"ms.{c} IS NOT DISTINCT FROM pat.{c}" for c in activeGrp)
            if activeGrp else "TRUE")
        pf = []
        for f in phi_miss:
            nm = f["name"]
            pf.append(f"CASE WHEN pat.{nm}_miss = 1 "
                      f"THEN COALESCE(ms.p_{nm}, 0.0) ELSE 1.0 END")
        prob_prod = " * ".join(pf) if pf else "1.0"
        prob_cond = f"({prob_prod})"
        prob_mass = f"{prob_cond} * COALESCE(ms.p_h, 0.0)"

        has_bounds = bool(iv_mode)
        bcpp = ""
        if iv_mode == "delta":
            rv = []
            for f in phi_miss:
                nm = f["name"]
                cp, dn = f"COALESCE(ms.p_{nm}, 0.0)", f"ms.den_{nm}"
                rv.append(f"CASE WHEN pat.{nm}_miss = 1 "
                          f"THEN ({_delta_relvar_sql(cp, dn, iv_alpha)}) "
                          f"ELSE 0 END")
            srv = " + ".join(rv)
            z = _z_critical(iv_alpha)
            bcpp = (
                f",\n    GREATEST(0, {prob_cond} - {z}*{prob_cond}"
                f"*SQRT(GREATEST(0,{srv})))::NUMERIC AS prob_lower"
                f",\n    LEAST(1, {prob_cond} + {z}*{prob_cond}"
                f"*SQRT(GREATEST(0,{srv})))::NUMERIC AS prob_upper")
        elif iv_mode in ("clt", "wilson", "hoeffding"):
            af = 1.0 - (1.0 - iv_alpha) ** (1.0 / max(len(phi_miss), 1))
            lp, hp = [], []
            for f in phi_miss:
                nm = f["name"]
                cp, dn = f"COALESCE(ms.p_{nm}, 0.0)", f"ms.den_{nm}"
                lo_s, hi_s = _interval_bounds_sql(iv_mode, cp, dn, af)
                lp.append(f"CASE WHEN pat.{nm}_miss = 1 "
                          f"THEN ({lo_s}) ELSE 1.0 END")
                hp.append(f"CASE WHEN pat.{nm}_miss = 1 "
                          f"THEN ({hi_s}) ELSE 1.0 END")
            bcpp = (f",\n    ({' * '.join(lp)})::NUMERIC AS prob_lower"
                    f",\n    ({' * '.join(hp)})::NUMERIC AS prob_upper")

        pp_cte = (f"pp AS (\n  SELECT pat.*,\n"
                  f"    {prob_cond}::NUMERIC AS prob_cond,\n"
                  f"    ({prob_mass})::NUMERIC AS probability{bcpp}\n"
                  f"  FROM patterns pat\n  LEFT JOIN ms ON {ms_join}\n)")

        ctes = [grouped, merged, mv_cte, pat_cte, pp_cte]
        iv_col = ""
        # 5) iv CTE: compute width from ms (theta-based H, no extra scan)
        iv_sql_post = None
        if has_bounds:
            rv_iv = []
            for f in phi_miss:
                nm = f["name"]
                rv_iv.append(
                    f"({_delta_relvar_sql(f'COALESCE(ms.p_{nm},0)', f'ms.den_{nm}', iv_alpha)})")
            z_iv = _z_critical(iv_alpha)
            srv_iv = " + ".join(rv_iv) if rv_iv else "0"
            iv_cte = (
                f"iv AS (\n  SELECT CASE WHEN SUM(ms.count_h) > 0\n"
                f"    THEN SUM(ms.count_h * LEAST(1.0, 2.0 * {z_iv}"
                f" * SQRT(GREATEST(0, {srv_iv}))))\n"
                f"      / SUM(ms.count_h)\n"
                f"    ELSE 0 END AS w\n  FROM ms\n)")
            ctes.append(iv_cte)
            iv_col = ", (SELECT w FROM iv)::NUMERIC AS _iv_w"

        # 6) Final SELECT (V1 structure)
        jc = ([f"mv.{c} IS NOT DISTINCT FROM pp.{c}" for c in activeGrp]
              + [f"mv.{i} = pp.{i}" for i in miss_ind])
        pp_join_mv = " AND ".join(jc) if jc else "TRUE"
        pay_outer = ", ".join(pay_cols)

        if has_bounds:
            cert_inner = (f"SELECT {pay_mv}, 1.0::NUMERIC AS score, "
                          f"1.0::NUMERIC AS lo, 1.0::NUMERIC AS hi, "
                          f"1.0::NUMERIC AS mass")
            unc_inner = (
                f"SELECT {pay_mv}, "
                f"COALESCE(pp.prob_cond,0)::NUMERIC AS score, "
                f"COALESCE(pp.prob_lower,0)::NUMERIC AS lo, "
                f"COALESCE(pp.prob_upper,1)::NUMERIC AS hi, "
                f"COALESCE(pp.probability,0)::NUMERIC AS mass")
        else:
            cert_inner = (f"SELECT {pay_mv}, 1.0::NUMERIC AS score, "
                          f"1.0::NUMERIC AS mass")
            unc_inner = (f"SELECT {pay_mv}, "
                         f"COALESCE(pp.prob_cond,0)::NUMERIC AS score, "
                         f"COALESCE(pp.probability,0)::NUMERIC AS mass")

        cert_cte = (f"__cert AS (\n  {cert_inner}\n"
                    f"  FROM mv WHERE mv.miss_sum = 0\n)")
        unc_cte = (f"__unc AS (\n  {unc_inner}\n"
                   f"  FROM mv\n  JOIN pp ON {pp_join_mv}\n"
                   f"  WHERE mv.miss_sum > 0\n)")
        rk_cte = (
            f"__rk AS (\n  SELECT u.*,\n"
            f"    SUM(u.mass) OVER (ORDER BY u.mass DESC, "
            f"{pay_outer}) AS cum_mass,\n"
            f"    SUM(u.mass) OVER () AS total_mass\n"
            f"  FROM __unc u\n)")
        ctes.extend([cert_cte, unc_cte, rk_cte])

        if has_bounds:
            c_sel = (f"SELECT {pay_outer}, MAX(score)::NUMERIC AS score, "
                     f"MAX(lo)::NUMERIC AS lo, MAX(hi)::NUMERIC AS hi, "
                     f"0::int AS mflag{iv_col}")
            u_sel = (f"SELECT {pay_outer}, MAX(score)::NUMERIC AS score, "
                     f"MIN(lo)::NUMERIC AS lo, MAX(hi)::NUMERIC AS hi, "
                     f"1::int AS mflag{iv_col}")
        else:
            c_sel = (f"SELECT {pay_outer}, MAX(score)::NUMERIC AS score, "
                     f"0::int AS mflag{iv_col}")
            u_sel = (f"SELECT {pay_outer}, MAX(score)::NUMERIC AS score, "
                     f"1::int AS mflag{iv_col}")

        final = (
            f"{c_sel}\nFROM __cert\nGROUP BY {pay_outer}\n"
            f"UNION ALL\n"
            f"{u_sel}\nFROM __rk\n"
            f"WHERE cum_mass <= total_mass * {fraction}::NUMERIC\n"
            f"   OR mass = (SELECT MAX(mass) FROM __rk)\n"
            f"GROUP BY {pay_outer}")

        pidx = list(range(len(pay_cols)))
        ng = len(activeGrp)
        sql = "WITH " + ",\n".join(ctes) + "\n" + final + ";"
        return sql, ng, len(pay_cols), pidx, has_bounds, iv_sql_post

    # ------------------------------------------------------------------
    def JoinQueryRewriter(self, fraction=0.5, iv_mode=None, iv_alpha=0.05):
        if not self.join_mode or not self.join_key:
            raise ValueError("JoinQueryRewriter needs join context.")
        T = self.T[:-4] if self.T.lower().endswith(".csv") else self.T
        S = self.S[:-4] if self.S.lower().endswith(".csv") else self.S
        kl = self.join_key
        join_src = f"{T} JOIN {S} USING ({', '.join(kl)})"
        combined_ord = {**(self.ordering_T or {}), **(self.ordering_S or {})}
        combined_miss = list(dict.fromkeys(
            (self.missing_T or []) + (self.missing_S or [])))
        combined_where = (self.where_T or []) + (self.where_S or [])
        joined_query = (f"SELECT {', '.join(self.select_cols)} FROM {join_src}"
                        + (f" WHERE {' AND '.join(combined_where)}"
                           if combined_where else ""))
        qr = QueryRewriterRanking(
            joined_query, combined_ord, combined_miss,
            score_threshold=getattr(self, "score_threshold", 0.0))
        qr.force_group_cols = list(kl)
        qr.select_cols = self.select_cols[:]
        return qr.groupLevelQueryRewriter(fraction, iv_mode, iv_alpha)


class QueryExecutorRanking(BaseQueryExecutor):
    FRACTION = 0.5
    _analyzed = set()

    def _composite_idx_analyze(self, table, ordering):
        t = table.strip('"')
        if t in QueryExecutorRanking._analyzed:
            return
        h = set()
        for cols in (ordering or {}).values():
            h.update(c.lower() for c in cols)
        if h:
            csv = ", ".join(f'"{c}"' for c in sorted(h))
            idx = f"idx_rk_{t}_{'_'.join(sorted(h))}"[:63]
            try:
                self.cur.execute(
                    f'CREATE INDEX IF NOT EXISTS "{idx}" ON "{t}" ({csv})')
                self.conn.commit()
            except Exception:
                try: self.conn.rollback()
                except Exception: pass
        try:
            self.cur.execute(f'ANALYZE "{t}"')
            self.conn.commit()
        except Exception:
            try: self.conn.rollback()
            except Exception: pass
        QueryExecutorRanking._analyzed.add(t)

    def _build_and_exec(self, base_query, ordering_T, missing_T,
                        ordering_S=None, missing_S=None,
                        join_key=None, score_threshold=0.0):
        sql1 = self._normalize_query(base_query.strip().rstrip(";"))
        wants_join = bool(
            re.search(r"\bJOIN\b.*\bUSING\s*\(", sql1, re.IGNORECASE))
        frac = getattr(self, "FRACTION", 0.5)
        iv_mode = getattr(self, "interval_mode", None)
        iv_alpha = getattr(self, "interval_alpha", 0.05)
        if wants_join:
            m = re.search(
                r"FROM\s+(?P<T>\S+)\s+JOIN\s+(?P<S>\S+)\s+USING\s*\(",
                sql1, re.IGNORECASE)
            tc = sc = None
            if m:
                Tn = m.group("T").strip('"')
                Sn = m.group("S").strip('"')
                tc, sc = self._table_columns(Tn), self._table_columns(Sn)
                self._ensure_support_indexes(sql1, [Tn, Sn])
                self._composite_idx_analyze(Tn, ordering_T)
                if ordering_S:
                    self._composite_idx_analyze(Sn, ordering_S)
            qr = QueryRewriterRanking(
                sql1, ordering_T, missing_T, ordering_S, missing_S,
                join_key, score_threshold, t_columns=tc, s_columns=sc)
            sql2, ng, np_, pidx, has_bounds, iv_sql = qr.JoinQueryRewriter(
                frac, iv_mode, iv_alpha)
        else:
            m = re.search(r"FROM\s+(\S+)", sql1, re.IGNORECASE)
            if m:
                self._ensure_support_indexes(sql1, [m.group(1)])
                self._composite_idx_analyze(m.group(1), ordering_T)
            qr = QueryRewriterRanking(
                sql1, ordering_T, missing_T,
                None, None, None, score_threshold)
            sql2, ng, np_, pidx, has_bounds, iv_sql = (
                qr.groupLevelQueryRewriter(frac, iv_mode, iv_alpha))
        _t0 = _time.perf_counter()
        self.cur.execute(sql2)
        rows = self.cur.fetchall()
        self._sql_elapsed = _time.perf_counter() - _t0
        iv_width = None
        if has_bounds and rows:
            iv_width = float(rows[0][-1] or 0)
        return rows, ng, np_, pidx, has_bounds, iv_width

    def run_flat(self, base_query, ordering_T, missing_T,
                 ordering_S=None, missing_S=None):
        """Returns (flat_rows, iv_width, elapsed, has_bounds)."""
        t0 = _time.perf_counter()
        rows, ng, np_, pidx, hb, iv_width = self._build_and_exec(
            base_query, ordering_T, missing_T, ordering_S, missing_S)
        elapsed = _time.perf_counter() - t0
        tail_n = 4 if hb else 2
        has_iv = hb and iv_width is not None
        out = []
        for r in rows:
            pay = tuple(r[i] for i in pidx)
            if has_iv:
                out.append(pay + tuple(r[-(tail_n+1):-1]))
            else:
                out.append(pay + tuple(r[-tail_n:]))
            if has_iv:
                out.append(pay + tuple(r[-(tail_n + 1):-1]))
            else:
                out.append(pay + tuple(r[-tail_n:]))
        return out, iv_width, elapsed, hb

    def run(self, base_query, ordering_T, missing_T,
            ordering_S=None, missing_S=None,
            join_key=None, score_threshold=0.0):
        rows, ng, np_, pidx, hb, iv_w = self._build_and_exec(
            base_query, ordering_T, missing_T, ordering_S, missing_S,
            join_key, score_threshold)
        if ng == 0 and np_ == 0:
            return rows
        if hb and iv_w is not None:
            rows = [r[:-1] for r in rows]
        return self._reshape(rows, ng, pidx, hb)

    @staticmethod
    def _reshape(rows, ng, pidx, has_bounds):
        groups = defaultdict(
            lambda: {"p": 0.0, "lo": 1.0, "hi": 1.0,
                     "miss": [], "cert": []})
        for row in rows:
            gk = tuple(row[:ng])
            pay = tuple(row[i] for i in pidx)
            if has_bounds:
                sc = float(row[-4] or 0)
                lo = float(row[-3] or 0)
                hi = float(row[-2] or 1)
                mf = int(row[-1] or 0)
            else:
                sc = float(row[-2] or 0)
                mf = int(row[-1] or 0)
                lo = hi = sc
            if sc > groups[gk]["p"]:
                groups[gk]["p"] = sc
            if lo < groups[gk]["lo"]:
                groups[gk]["lo"] = lo
            if hi > groups[gk]["hi"]:
                groups[gk]["hi"] = hi
            (groups[gk]["cert"] if mf == 0
             else groups[gk]["miss"]).append(pay)
        out = []
        for gk, d in groups.items():
            if has_bounds:
                out.append(tuple(
                    list(gk) + [d["p"], d["lo"], d["hi"],
                                d["miss"] or None, d["cert"] or None]))
            else:
                out.append(tuple(
                    list(gk) + [d["p"], d["miss"] or None,
                                d["cert"] or None]))
        return out
