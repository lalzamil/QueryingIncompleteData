import os, json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
from causallearn.search.ConstraintBased.PC import pc
from minimalSeparatorsSearch import minimalSeparatorsSearch
class DataDAGs:
    def __init__(self,in_path,out_csv="",out_json="",out_png_regular=""):
        # pass
        # rng = np.random.default_rng(13)
        self.rng = np.random.default_rng()
        self.in_path=in_path
        self.out_csv=out_csv
        self.out_json=out_json
        self.out_png_regular=out_png_regular
        self.df = pd.read_csv(self.in_path)
        self.cols=self.df.columns.tolist()

## this part uses the PC algorthim to discover the dag of the input data ((part1 of data experements))
    def _drop_identifier_like(self, df):
        """ drop ID-like columns before discovery"""
        close2_unique_ratio=0.97  ## nyc dataset was exploding so im dorping near uique cols before discoevry
        max_raw_card=500
        cand = []
        for c in df.columns:
            if df[c].nunique() == len(df):           # all unique values
                cand.append(c)
            if df[c].nunique() / max(1, len(df)) >= close2_unique_ratio:
                cand.append(c)
            if (not pd.api.types.is_numeric_dtype(df[c])) and df[c].nunique() > max_raw_card:
                cand.append(c)
            if str(c).lower() in {"id", "uid", "guid"}:
                cand.append(c)
        cand = list(dict.fromkeys(cand))
        return df.drop(columns=cand), cand

    def discover_dag_and_plot(
        self,
        alpha: float = 0.05,
        orient_undirected: bool = True,
        bins_per_numeric: int = 5,
        verbose: bool = False,
        saveCompleteDiscovered: bool = False,
    ):
        """
        1) PC -> CPDAG
        2) Extract parents from directed edges
        3) Optionally orient undirected edges along a total order to get an acyclic DAG
        4) Save JSON and plot layered DAG
        """
        df0 = self.df.copy()
        df, dropped = self._drop_identifier_like(df0)
        names = list(df.columns)

        # --- choose CI test & prep X ---
        def all_numeric(d):
            return all(pd.api.types.is_numeric_dtype(d[c]) for c in d.columns)
        def all_categ(d):
            return all(not pd.api.types.is_numeric_dtype(d[c]) for c in d.columns)

        if all_numeric(df):
            indep_test = "fisherz"
            X = df.to_numpy(dtype=float)
        elif all_categ(df):
            indep_test = "gsq"
            X = np.column_stack([pd.Categorical(df[c]).codes for c in names]).astype(int)
        else:
            indep_test = "gsq"  # robust fallback: discretize numerics
            d2 = df.copy()
            for c in names:
                if pd.api.types.is_numeric_dtype(d2[c]):
                    q = min(bins_per_numeric, max(1, d2[c].nunique()))
                    d2[c] = pd.qcut(d2[c].rank(method="first"), q=q, duplicates="drop").cat.codes
                else:
                    d2[c] = pd.Categorical(d2[c]).codes
            X = d2.to_numpy(dtype=int)

        # --- run PC (CPDAG) ---
        cg = pc(X, alpha=alpha, indep_test=indep_test, stable=True, verbose=verbose)
        A = cg.G.graph
        p = len(names)

        # Decode edges
        directed, undirected, bidirected = [], [], []
        for i in range(p):
            for j in range(i + 1, p):
                a, b = A[i, j], A[j, i]
                if a == 1 and b == -1:
                    directed.append((names[i], names[j]))      # i -> j
                elif a == -1 and b == 1:
                    directed.append((names[j], names[i]))      # j -> i
                elif a == -1 and b == -1:
                    undirected.append((names[i], names[j]))    # i -- j
                elif a == 1 and b == 1:
                    bidirected.append((names[i], names[j]))    # i <-> j

        # Parents from directed edges
        parents = {v: [] for v in names}
        for u, v in directed:
            if u != v:
                parents[v].append(u)

        # Topo sort (Kahn)
        def topo_sort(nodes, edges):
            adj = {n: [] for n in nodes}
            indeg = {n: 0 for n in nodes}
            for u, v in edges:
                if u == v:
                    continue
                adj[u].append(v)
                indeg[v] += 1
            from collections import deque
            Q = deque([n for n in nodes if indeg[n] == 0])
            order, seen = [], set(Q)
            while Q:
                x = Q.popleft()
                order.append(x)
                for y in adj[x]:
                    indeg[y] -= 1
                    if indeg[y] == 0 and y not in seen:
                        Q.append(y); seen.add(y)
            # append leftovers (if directed part had cycles)
            for n in nodes:
                if n not in order:
                    order.append(n)
            return order

        topo = topo_sort(names, directed)

        # Orient undirected edges along topo (consistent extension)
        if orient_undirected and undirected:
            pos = {n: i for i, n in enumerate(topo)}
            dag_edges = directed[:]
            for u, v in undirected:
                dag_edges.append((u, v) if pos[u] <= pos[v] else (v, u))
            # rebuild parents and topo
            parents = {n: [] for n in names}
            for u, v in dag_edges:
                if u != v:
                    parents[v].append(u)
            topo = topo_sort(names, dag_edges)

        # **Cycle guard**: remove any accidental self-loops and drop back-edges
        for v, pas in list(parents.items()):
            parents[v] = [u for u in pas if u != v]
        # If you want to be extra-safe, drop any parent u that appears after v in topo:
        pos = {n: i for i, n in enumerate(topo)}
        for v, pas in parents.items():
            parents[v] = [u for u in pas if pos[u] < pos[v]]

        # Save JSON
        payload = {
            # "ci_test": indep_test,
            # "alpha": alpha,
            "dropped_identifier_like": dropped,
            "parents": {k: list(v) for k, v in parents.items()},
            "topo_order": topo,
            # "directed_edges": directed,
            # "undirected_edges": undirected,
            # "bidirected_edges": bidirected,
        }
        if saveCompleteDiscovered:
            # self.out_json= "ExpPlan/Sep25_THUR/bank_discovered_dag.json"
            print(" the outjson is: ----------")
            print(self.out_json)
            self.out_json= self.out_json.replace("mnar1_dag", "mnar1_discovered_dag")
            print(" the outjson is: ----------")
            print(self.out_json)
        if self.out_json and saveCompleteDiscovered:
            os.makedirs(os.path.dirname(self.out_json) or ".", exist_ok=True)
            with open(self.out_json, "w") as f:
                json.dump(payload, f, indent=2)

        # Plot
        if self.out_json:
          self.plot_dag_layered(parents, self.out_png_regular)

        return parents, topo, directed

    def plot_dag_layered(self,
        parents, save_path,
        layer_gap=1.4,        # smaller = layers closer
        y_gap=1.1,            # smaller = nodes within a layer closer
        pad_x=0.6, pad_y=0.4, # frame padding so nothing is cut
        node_size=520,        # scatter size (points^2)
        node_linewidth=1.8,
        font_size=12,
        edge_lw=1.8,
        arrow_mutation=10,    # arrow head size
        dpi=200
    ):


        nodes = list(parents.keys())

        @lru_cache(maxsize=None)
        def depth(u):
            pas = parents.get(u, [])
            if not pas:
                return 0
            return 1 + max(depth(p) for p in pas)

        # layered positions
        levels = {}
        for u in nodes:
            d = depth(u)
            levels.setdefault(d, []).append(u)
        for d in levels:
            levels[d] = sorted(levels[d])

        pos = {}
        for layer in sorted(levels):
            items = levels[layer]
            k = len(items)
            ys = (np.arange(k) - (k - 1) / 2.0) * y_gap
            xs = np.full(k, layer * layer_gap)
            for i, node in enumerate(items):
                pos[node] = (xs[i], ys[i])

        xs_all = np.array([pos[n][0] for n in nodes])
        ys_all = np.array([pos[n][1] for n in nodes])
        span_x = (xs_all.max() - xs_all.min()) + 2 * pad_x
        span_y = (ys_all.max() - ys_all.min()) + 2 * pad_y

        fig, ax = plt.subplots(figsize=(max(4, 1.1 * span_x), max(3, 1.1 * span_y)))

        # --- compute shrink from node size so edges are SHORT ---
        # scatter uses area (points^2); radius in points:
        radius_pts = (node_size / np.pi) ** 0.5
        shrink = int(radius_pts) + 2  # a touch more so arrows don't touch circles

        # --- edges ---
        for child, pas in parents.items():
            x2, y2 = pos[child]
            for p in pas:
                x1, y1 = pos[p]
                ax.annotate(
                    "",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>", lw=edge_lw,
                        shrinkA=shrink, shrinkB=shrink,  # << shorter edges
                        mutation_scale=arrow_mutation
                    ),
                    clip_on=False
                )

        # --- hollow nodes + readable labels ---
        for node, (x, y) in pos.items():
            ax.scatter([x], [y], s=node_size, facecolors="none",
                    edgecolors="black", linewidths=node_linewidth, zorder=3)
            ax.text(
                x, y + 0.22, node, fontsize=font_size,
                ha="center", va="bottom", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.3)
            )

        # frame & export
        ax.set_xlim(xs_all.min() - pad_x, xs_all.max() + pad_x)
        ax.set_ylim(ys_all.min() - pad_y, ys_all.max() + pad_y)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
        plt.show()
        plt.close(fig)


# print("======================================")
# print(" dicovering the dag of comelete input")
# # in_path = "dominance_test_complete.csv" ## indep nodes
# # in_path = "z_dagTests/dominance_test_complete.csv"
# in_path = "rwDatasets/bank_complete.csv"
# out_json = "z_dagTests/bank_discovered_dag.json"
# out_png_regular = "z_dagTests/bank_discovered_dag.png"

# data_dag = DataDAGs(in_path=in_path,out_json=out_json,out_png_regular=out_png_regular)
# _,_,_=data_dag.discover_dag_and_plot()




class logisticInject:
    def __init__(self):
        pass

    def standard_z(self, x):
        x = pd.Series(x, copy=False)
        if not pd.api.types.is_numeric_dtype(x):
            x = x.astype("category").cat.codes
        x = x.astype(float)
        s = x.std(ddof=0)
        return (x - x.mean()) / (s if s and np.isfinite(s) else 1.0)

    def sigmoid(self, z): #maps real numbers to probabilities
        return 1/(1+np.exp(-z))

    def calibrate_intercept(self, eta, target, lo=-20, hi=20, iters=40):
        target = float(target)
        a_lo, a_hi = lo, hi
        for _ in range(iters):
            a_mid = 0.5*(a_lo+a_hi)
            m = self.sigmoid(a_mid + eta).mean()
            if m < target: a_lo = a_mid
            else:          a_hi = a_mid
        return 0.5*(a_lo+a_hi)

    def make_dir(self, path):
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)


class MnarInjector:
    def __init__(self, in_path, out_csv, out_json, out_png_regular, seed=None):
        self.in_path = in_path
        self.out_csv = out_csv
        self.out_json = out_json
        self.out_png_regular = out_png_regular

        self.df_full = pd.read_csv(self.in_path)  # pristine (complete)
        self.df = self.df_full.copy()             # will be modified with NaNs
        self.cols = self.df.columns.tolist()

        self.rng = np.random.default_rng(seed)
        self.logistic_inject = logisticInject()

        # for plotting/json files
        # r_nodes: "R_col" -> {"type": "MCAR/MAR/MNAR", "causes": [...], "rate": float}
        self.r_nodes = {}
        self.partially_observed = set()

        self.factorization = {"order": [], "X_sets": {}}  # {"order":["income","bonus"], "X_sets":{"income":["education"], "bonus":["job"]}}


        # self.dag_discovery=DataDAGs(in_path=self.in_path,out_json="z_dagTests/bank_mnar1_mgraphInjectorClass.json",
        #                             out_png_regular="z_dagTests/bank_mnar1_mgraphInjector.png")
        self.dag_discovery=DataDAGs(in_path=self.in_path,out_json=self.out_json,
                                    out_png_regular=self.out_png_regular)



    def plan_injection(self, alpha=0.05, bins_per_numeric=5, split_seed=None, mcar_rate=0.15, numeric_only_mnar=True,min_mnar_targets=3, relax_on_empty=True):
                """Discover once , pick the same columns once."""
                parents_pc, topo, _ = self.dag_discovery.discover_dag_and_plot(
                    alpha=alpha, bins_per_numeric=bins_per_numeric, saveCompleteDiscovered=True
                )
                universe = set(topo)

                mcar_cols, mar_parents_pool, mar_targets, mnar_targets = self.split_list_randomly(
                    seed=split_seed,
                    universe=universe,                       # align with PC universe
                    numeric_only_mnar=numeric_only_mnar,
                    min_mnar_targets=min_mnar_targets
                )

                # If split couldn’t satisfy min_mnar_targets, bail with a clear note
                if not mnar_targets:
                    return {"note": "Not enough MNAR candidates in the discovery universe to satisfy "
                                    f"min_mnar_targets={min_mnar_targets} with "
                                    f"numeric_only_mnar={numeric_only_mnar}."}

                # Stable order
                mnar_targets_ordered = self.align_list_to_order(mnar_targets, list(universe))
                if not mnar_targets_ordered:
                    return {"note": "No MNAR targets available after alignment (unexpected since we picked from universe)."}

                # Pick first MCAR parent not colliding with MNAR
                first_mcar_parents = []
                for c in mcar_cols:
                    if c not in mnar_targets_ordered:
                        first_mcar_parents = [c]
                        break

                first_mcar_rates = {p: mcar_rate for p in first_mcar_parents}
                remaining_mcar   = [c for c in mcar_cols if c not in first_mcar_parents]

                return {
                    "topo": list(universe),
                    "mnar_targets_ordered": mnar_targets_ordered,
                    "mar_parents_pool": mar_parents_pool,
                    "mar_targets": mar_targets,
                    "first_mcar_parents": first_mcar_parents,
                    "first_mcar_rates": first_mcar_rates,
                    "remaining_mcar": remaining_mcar,
                    "mcar_rate": mcar_rate
                }

    def run_plan(self, plan, rate_mnar=0.10, mar_rate=0.25, beta_mar=0.8, beta_prev=1.0, rng_seed=None):
        """Execute the frozen plan with a chosen MNAR rate."""
        if rng_seed is not None:
            self.rng = np.random.default_rng(rng_seed)  # reproducible masks

        # MNAR (same targets/parents every time; only rate changes)
        self.inject_mnar_type_1_ordered(
            targets=plan["mnar_targets_ordered"],
            mar_parents=plan["mar_parents_pool"],
            first_mcar_parents=plan["first_mcar_parents"],
            first_mcar_rates=plan["first_mcar_rates"],
            rate=rate_mnar,
            beta_mar=beta_mar,
            beta_prev=beta_prev
        )

        # MAR (keep parents/targets fixed)
        if plan["mar_targets"]:
            self.inject_mar_columns(
                plan["mar_targets"],
                parents=plan["mar_parents_pool"],
                rate=mar_rate,
                alpha_par=1.0
            )

        # MCAR (keep same columns; same rate)
        if plan["remaining_mcar"]:
            self.inject_mcar_columns(plan["remaining_mcar"], rate=plan["mcar_rate"])




    # ---------- utilities ----------
    def drop_identifier_like(self):
        df = self.df.copy()
        close2_unique_ratio=0.97  ## nyc dataset was exploding so im dorping near uique cols before discoevry
        max_raw_card=500
        cand = []
        for c in df.columns:
            if df[c].nunique() == len(df):           # all unique values
                cand.append(c)
            if df[c].nunique() / max(1, len(df)) >= close2_unique_ratio:
                cand.append(c)
            if (not pd.api.types.is_numeric_dtype(df[c])) and df[c].nunique() > max_raw_card:
                cand.append(c)
            if str(c).lower() in {"id", "uid", "guid"}:
                cand.append(c)
        cand = list(dict.fromkeys(cand))
        return df.drop(columns=cand), cand
        # cand = []
        # for c in df.columns:
        #     if df[c].nunique() == len(df) or str(c).lower() in {"id", "uid", "guid"}:
        #         cand.append(c)
        # cand = list(dict.fromkeys(cand))
        # return df.drop(columns=cand), cand

    # def split_list_randomly(self, num_splits=4, seed=None):
    #     df_without_id, _ = self.drop_identifier_like()
    #     cols =  df_without_id.columns.tolist()
    #     shuffler = random.Random(seed) if seed is not None else random
    #     cols = cols[:]
    #     shuffler.shuffle(cols)

    #     sublist_size= len(cols) // num_splits
    #     reminder = len(cols) % num_splits
    #     subs = []
    #     idx=0
    #     for j in range(num_splits):
    #         if j < reminder:
    #              sz = sublist_size + 1
    #         else:
    #             sz = sublist_size
    #         subs.append(cols[idx:idx+sz])
    #         idx += sz
    #     while len(subs) < 4: subs.append([])
    #     # order: [MCAR_targets, MAR_parents, MAR_targets, MNAR_targets]
    #     return subs[:4]
    #     # return subs


    def split_list_randomly(
        self,
        num_splits: int = 4,
        seed: int | None = None,
        universe: set[str] | list[str] | None = None,
        numeric_only_mnar: bool = False,
        min_mnar_targets: int = 1,
        numericish_thresh: float = 0.95,
    ):
        """
        Returns [MCAR_targets, MAR_parents_pool, MAR_targets, MNAR_targets].
        MNAR is picked FIRST from the appropriate pool:
        - numeric_only_mnar=True  -> numeric-ish columns only
        - numeric_only_mnar=False -> any column
        Then the remainder is randomly split into MCAR / MAR-parents / MAR-targets.
        """
        df_wo_id, _ = self.drop_identifier_like()
        cols_all = list(df_wo_id.columns)

        # Intersect with discovery universe (if provided)
        if universe is not None:
            U = set(universe)
            cols_all = [c for c in cols_all if c in U]

        def is_numericish(col: str) -> bool:
            s = self.df[col]
            if pd.api.types.is_numeric_dtype(s):
                return True
            if pd.api.types.is_object_dtype(s):
                coerced = pd.to_numeric(s, errors="coerce")
                return coerced.notna().mean() >= numericish_thresh
            return False

        rng = random.Random(seed)

        # --- MNAR pool based on the flag
        if numeric_only_mnar:
            mnar_pool = [c for c in cols_all if is_numericish(c)]
        else:
            mnar_pool = cols_all[:]

        # Need at least 1 element in the MNAR pool to satisfy min_mnar_targets
        if len(mnar_pool) == 0:
            return [], [], [], []  # caller will handle with a clear note

        # Pick MNAR first (random, reproducible)
        take = min(min_mnar_targets, len(mnar_pool))
        mnar_pool_shuf = mnar_pool[:]
        rng.shuffle(mnar_pool_shuf)
        mnar_targets = mnar_pool_shuf[:take]

        # Remainder pool
        used = set(mnar_targets)
        rest = [c for c in cols_all if c not in used]
        rng.shuffle(rest)

        # Split remainder into MCAR / MAR-parents / MAR-targets (3 buckets)
        if num_splits < 4:
            num_splits = 4
        buckets_needed = 3
        base = len(rest) // buckets_needed
        rem = len(rest) %  buckets_needed

        buckets = []
        idx = 0
        for j in range(buckets_needed):
            sz = base + (1 if j < rem else 0)
            buckets.append(rest[idx: idx+sz])
            idx += sz

        mcar_cols        = buckets[0] if len(buckets) > 0 else []
        mar_parents_pool = buckets[1] if len(buckets) > 1 else []
        mar_targets      = buckets[2] if len(buckets) > 2 else []

        return [mcar_cols, mar_parents_pool, mar_targets, mnar_targets]



    def align_list_to_order(self, items, topo_order):
        pos = {n: i for i, n in enumerate(topo_order)}
        return sorted([x for x in items if x in pos], key=lambda x: pos[x])

    # ---------- injectors ----------
    def inject_mcar_columns(self, cols, rate=0.2):
        n = len(self.df)
        for c in cols:
            if c not in self.df.columns: continue
            mask = self.rng.random(n) < rate
            self.df.loc[mask, c] = np.nan
            self.partially_observed.add(c)
            self.r_nodes[f"R_{c}"] = {"type": "MCAR", "causes": [], "rate": float(mask.mean())}

    def inject_mar_columns(self, targets, parents, rate=0.3, alpha_par=1.0, alpha_0_extra=0.0):
        if isinstance(targets, str): targets = [targets]
        parents = [] if parents is None else list(parents)
        for t in targets:
            if t not in self.df.columns: continue
            eta = 0.0
            for p in parents:
                if p not in self.df_full.columns: continue
                eta = eta + alpha_par * self.logistic_inject.standard_z(self.df_full[p])
            eta = np.asarray(eta, dtype=float)
            a0 = self.logistic_inject.calibrate_intercept(eta, rate) + alpha_0_extra
            prob = self.logistic_inject.sigmoid(a0 + eta)
            mask = self.rng.random(len(prob)) < prob
            self.df.loc[mask, t] = np.nan
            self.partially_observed.add(t)
            self.r_nodes[f"R_{t}"] = {"type": "MAR", "causes": list(parents), "rate": float(mask.mean())}

    def inject_mnar_type_1_ordered(self, targets, mar_parents=None,
                                   first_mcar_parents=None, first_mcar_rates=None,
                                   rate=0.30, beta_mar=1.0, beta_prev=1.0, alpha0_extra=0.0):
    # def inject_mnar_type_1_ordered(self, targets, mar_parents=None,
    #                                first_mcar_parents=None, first_mcar_rates=None,
    #                                rates=[0.05,0.1,0.20], beta_mar=1.0, beta_prev=1.0, alpha0_extra=0.0):
        if isinstance(targets, str):
            targets = [targets]

        self.factorization["order"] = list(targets)
        k = len(targets)
        if k == 0: return []
        mar_parents = [] if mar_parents is None else list(mar_parents)
        first_mcar_par = [] if first_mcar_parents is None else list(first_mcar_parents)
        first_mcar_rates = first_mcar_rates or {}

        # j=1 (no self-masking; depends on mar_parents ∪ first_mcar_par)
        Y1 = targets[0]
        X1 = list(dict.fromkeys((mar_parents or []) + (first_mcar_par or [])))

        eta = 0.0
        for pcol in mar_parents:
            if pcol in self.df_full.columns:
                eta = eta + beta_mar * self.logistic_inject.standard_z(self.df_full[pcol])
        for pcol in first_mcar_par:
            if pcol in self.df_full.columns:
                eta = eta + beta_mar * self.logistic_inject.standard_z(self.df_full[pcol])
        eta = np.asarray(eta, dtype=float)
        a0 = self.logistic_inject.calibrate_intercept(eta, rate) + alpha0_extra
        prob = self.logistic_inject.sigmoid(a0 + eta)
        R = self.rng.random(len(prob)) < prob
        self.df.loc[R, Y1] = np.nan
        self.partially_observed.add(Y1)
        self.r_nodes[f"R_{Y1}"] = {"type": "MNAR", "causes": mar_parents + first_mcar_par, "rate": float(R.mean())}

        self.factorization["X_sets"][Y1] = X1

        # now MCAR-hide first_mcar_par (to make mechanism MNAR w.r.t observed data)
        for pcol in first_mcar_par:
            if pcol in self.df.columns:
                pr = float(first_mcar_rates.get(pcol, 0.15))
                mask_p = self.rng.random(len(self.df)) < pr
                self.df.loc[mask_p, pcol] = np.nan
                self.partially_observed.add(pcol)
                self.r_nodes[f"R_{pcol}"] = {"type": "MCAR", "causes": [], "rate": float(mask_p.mean())}

        # j>=2 (no self-masking; depends on mar_parents ∪ previous Y's)
        for j in range(1, k):
            Yj = targets[j]
            Xj = list(dict.fromkeys((mar_parents or []) + targets[:j]))

            eta = 0.0
            for pcol in mar_parents:
                if pcol in self.df_full.columns:
                    eta = eta + beta_mar * self.logistic_inject.standard_z(self.df_full[pcol])
            for prev in targets[:j]:
                eta = eta + beta_prev * self.logistic_inject.standard_z(self.df_full[prev])
            eta = np.asarray(eta, dtype=float)
            a0 = self.logistic_inject.calibrate_intercept(eta, rate) + alpha0_extra
            prob = self.logistic_inject.sigmoid(a0 + eta)
            R = self.rng.random(len(prob)) < prob
            self.df.loc[R, Yj] = np.nan
            self.partially_observed.add(Yj)
            self.r_nodes[f"R_{Yj}"] = {
                "type": "MNAR", "causes": mar_parents + targets[:j], "rate": float(R.mean())
            }
            self.factorization["X_sets"][Yj] = Xj

    # ---------- pipeline that picks a random valid order, splits, injects ----------
    def mnar1_random_order_pipeline(
        self, alpha=0.05, bins_per_numeric=5, seed=None, split_seed=None,
        mcar_rate=0.15, rate_mnar=0.30, beta_mar=0.8, beta_prev=1.0):
        # parents_pc, topo, directed, undirected, dag_edges = self.discovery_pc(alpha=alpha, bins_per_numeric=bins_per_numeric, seed=seed)

        parents_pc, topo, directed  = self.dag_discovery.discover_dag_and_plot(alpha=alpha, bins_per_numeric=bins_per_numeric,saveCompleteDiscovered=True)

        mcar_cols, mar_parents_pool, mar_targets, mnar_targets = self.split_list_randomly(seed=split_seed)
        mnar_targets_ordered = self.align_list_to_order(mnar_targets, topo)
        if not mnar_targets_ordered:
            return {"note": "No MNAR targets selected by split; adjust split."}

        first_mcar_parents = [mcar_cols[0]] if mcar_cols else []
        first_mcar_rates = {p: mcar_rate for p in first_mcar_parents}

        self.inject_mnar_type_1_ordered(
            targets=mnar_targets_ordered,
            mar_parents=mar_parents_pool,
            first_mcar_parents=first_mcar_parents,
            first_mcar_rates=first_mcar_rates,
            rate=rate_mnar,
            beta_mar=beta_mar,
            beta_prev=beta_prev
        )

        if mar_targets:
            self.inject_mar_columns(mar_targets, parents=mar_parents_pool, rate=0.25, alpha_par=1.0)

        remaining_mcar = [c for c in mcar_cols if c not in first_mcar_parents]
        if remaining_mcar:
            self.inject_mcar_columns(remaining_mcar, rate=mcar_rate)

        return {
            "random_topo_order": topo,
            "mcar_cols": mcar_cols,
            "mar_parents_pool": mar_parents_pool,
            "mar_targets": mar_targets,
            "mnar_targets_ordered": mnar_targets_ordered
        }

    # ---------- SAVE RESULTS ---------- ---
    def save_outputs(self):
        if self.out_csv:
            self.logistic_inject.make_dir(self.out_csv)
            self.df.to_csv(self.out_csv, index=False)
        if self.out_json:
            self.logistic_inject.make_dir(self.out_json)
            print(" ---->>>i am here and the json file name is --->>> ", self.out_json)
            with open(self.out_json, "w") as f:
                json.dump({
                    "in_path": self.in_path,
                    "out_csv": self.out_csv,
                    "mechanisms": self.r_nodes,
                    "partially_observed": sorted(self.partially_observed)
                }, f, indent=2)









    # ---------- PLOT m-graph (mechanism edges only; requested colors; hollow where fully observed) ----------
    def plot_m_graph(self, alpha=0.05, bins_per_numeric=5, dpi=200):
        """
        Layered m-graph:
        • Black edges: data dependencies discovered by PC on COMPLETE data
        • Red edges: mechanism edges (cause -> R_node)
        • Nodes:
            - Data with missingness: FILLED & bold by type {MCAR=green, MAR=pink, MNAR=orange}
            - Data fully observed: hollow (no fill)
            - Mechanism nodes R_*: hollow (no fill)
        Saves to self.out_png_regular.
        """
        if not self.out_png_regular:
            return

        # 1) Data-DAG from complete table (for layering)
        # parents_data, topo, directed, undirected, _= self.discovery_pc(alpha=alpha, bins_per_numeric=bins_per_numeric)
        print(">>>>>> in m-graph before discovery<<<<<<<<<<")
        parents_data, topo, directed= self.dag_discovery.discover_dag_and_plot(alpha=alpha, bins_per_numeric=bins_per_numeric)


        # 2) Map each DATA column -> mechanism type if it was injected
        data_type = {}  # e.g., {"income":"MNAR", "education":"MCAR", ...}
        for R, meta in self.r_nodes.items():
            if R.startswith("R_"):
                tgt = R[2:]
                if tgt in self.df_full.columns:
                    data_type[tgt] = meta.get("type", "MNAR")

        # 3) Build a combined parent map for layout (data edges only for depths)
        data_nodes = list(self.df_full.columns)
        parents_layout = {n: list(parents_data.get(n, [])) for n in data_nodes}

        # 4) Cycle-safe depths for layered layout (data nodes only)
        depths, temp = {}, set()
        def depth(u):
            if u in depths: return depths[u]
            if u in temp:   depths[u] = 0; return 0  # break cycles if any
            temp.add(u)
            best = 0
            for p in parents_layout.get(u, []):
                best = max(best, depth(p) + 1)
            temp.remove(u)
            depths[u] = best
            return best
        for u in data_nodes:
            depth(u)

        # 5) Add mechanism nodes one layer below their target (or max(causes)+1)
        all_nodes = set(data_nodes)
        for R, meta in self.r_nodes.items():
            all_nodes.add(R)
            # prefer target name after "R_"
            if R.startswith("R_") and R[2:] in depths:
                depths[R] = depths[R[2:]] + 1
            else:
                # fall back to cause-based placement
                cause_layers = [depths[c] for c in meta.get("causes", []) if c in depths]
                depths[R] = (max(cause_layers) + 1) if cause_layers else 0

        # 6) Group by layer and position
        levels = {}
        for u in all_nodes:
            levels.setdefault(depths.get(u, 0), []).append(u)
        # within a layer: show data first, then R_* for readability
        for d in levels:
            levels[d] = sorted(levels[d], key=lambda v: (v.startswith("R_"), v))

        layer_gap, y_gap = 1.4, 1.05
        pos = {}
        for layer in sorted(levels):
            items = levels[layer]
            k = len(items)
            ys = (np.arange(k) - (k - 1) / 2.0) * y_gap
            xs = np.full(k, layer * layer_gap)
            for i, node in enumerate(items):
                pos[node] = (xs[i], ys[i])

        # 7) Set up figure
        import matplotlib.pyplot as plt
        xs_all = np.array([pos[n][0] for n in pos])
        ys_all = np.array([pos[n][1] for n in pos])
        pad_x, pad_y = 0.6, 0.4
        fig, ax = plt.subplots(
            figsize=(max(4, 1.1*((xs_all.max()-xs_all.min()) + 2*pad_x)),
                    max(3, 1.1*((ys_all.max()-ys_all.min()) + 2*pad_y)))
        )

        node_size = 540
        radius_pts = (node_size/np.pi)**0.5
        shrink = int(radius_pts) + 2

        # 8) Draw DATA edges (black) using directed list from PC
        for u, v in directed:
            if (u in pos) and (v in pos):
                x1, y1 = pos[u]; x2, y2 = pos[v]
                ax.annotate(
                    "", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4,
                                    shrinkA=shrink, shrinkB=shrink,
                                    mutation_scale=10),
                    color="black", clip_on=False
                )

        # 9) Draw MECHANISM edges (red) from causes → R_*
        for R, meta in self.r_nodes.items():
            for c in meta.get("causes", []):
                if (c in pos) and (R in pos):
                    x1, y1 = pos[c]; x2, y2 = pos[R]
                    ax.annotate(
                        "", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", lw=2.0,
                                        shrinkA=shrink, shrinkB=shrink,
                                        mutation_scale=11),
                        color="#eb1515ff", clip_on=False
                    )

        # 10) Draw nodes with requested styles
        color_map = {"MCAR": "#2ca02c", "MAR": "#e377c2", "MNAR": "#ff7f0e"}  # green / pink / orange
        for n, (x, y) in pos.items():
            if n.startswith("R_"):  # mechanism nodes → HOLLOW
                ax.scatter([x], [y], s=node_size, facecolors="none",
                        edgecolors="black", linewidths=1.8, zorder=3)
            else:
                if n in self.partially_observed:
                    t = data_type.get(n, "MNAR")
                    ax.scatter([x], [y], s=node_size, facecolors=color_map.get(t, "#ff7f0e"),
                            edgecolors="black", linewidths=2.2, zorder=3)  # bold & colored
                else:
                    ax.scatter([x], [y], s=node_size, facecolors="none",
                            edgecolors="black", linewidths=1.8, zorder=3)  # fully observed → hollow
            # label
            ax.text(x, y + 0.22, n, fontsize=11, ha="center", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.25), zorder=4)

        # 11) Legend (matches requested semantics)
        ax.scatter([], [], s=node_size, facecolors="none", edgecolors="black", linewidths=1.8,
                label="Fully observed ")
        ax.scatter([], [], s=node_size, facecolors="#2ca02c", edgecolors="black", linewidths=2.2,
                label="MCAR")
        ax.scatter([], [], s=node_size, facecolors="#e377c2", edgecolors="black", linewidths=2.2,
                label="MAR ")
        ax.scatter([], [], s=node_size, facecolors="#ff7f0e", edgecolors="black", linewidths=2.2,
                label="MNAR ")
        # ax.scatter([], [], s=node_size, facecolors="none", edgecolors="black", linewidths=1.8,label="Mechanism node R_* (hollow)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

        # 12) Finish
        ax.set_xlim(xs_all.min() - pad_x, xs_all.max() + pad_x)
        ax.set_ylim(ys_all.min() - pad_y, ys_all.max() + pad_y)
        ax.set_axis_off()
        self.logistic_inject.make_dir(self.out_png_regular)
        fig.tight_layout()
        fig.savefig(self.out_png_regular, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)






# inj = MnarInjector(
#     in_path="rwDatasets/bank_complete.csv",
#     out_csv="z_dagTests/bank_mnar1_injected.csv",
#     out_json="z_dagTests/bank_mnar1_dag.json",
#     out_png_regular="z_dagTests/bank_mnar1_mgraph.png",
#     # seed=7
# )



# inj = MnarInjector(
#     in_path="rwDatasets/bank_complete.csv",
#     out_csv="ExpPlan/Sep25_THUR/bank_mnar1_injected.csv",
#     out_json="ExpPlan/Sep25_THUR/bank_mnar1_dag.json",
#     out_png_regular="ExpPlan/Sep25_THUR/bank_mnar1_mgraph.png",
#     # seed=7
# )

# # 1) Random order >> split >> inject MNAR-1 (ordered), plus MAR/MCAR as configured
# summary = inj.mnar1_random_order_pipeline(
#     alpha=0.05, bins_per_numeric=5,
#     #   seed=123, split_seed=999,
#     mcar_rate=0.15, rate_mnar=0.30, beta_mar=0.8, beta_prev=1.0
# )

# # 2) Save files
# inj.save_outputs()        # writes CSV + JSON
# inj.plot_m_graph()        # writes PNG

# print("Pipeline summary:", summary)
# print("Saved:", inj.out_csv, inj.out_json, inj.out_png_regular)


# discovered_complete_json = "ExpPlan/Sep25_THUR/bank_discovered_dag.json"
# mechamic_json = "ExpPlan/Sep25_THUR/bank_mnar1_dag.json"
# seprators_out_csv ="ExpPlan/Sep25_THUR/ban_mnar1_minimal_separators_Xi.csv"

# find_serparators_inJson = minimalSeparatorsSearch(discovered_complete_json,mechamic_json,seprators_out_csv)

# find_serparators_inJson.findMySeparators()



complete_datasets_names = [
    # "rwDatasets/bank_complete.csv",
    "rwDatasets/nyc_complete.csv",
    # "rwDatasets/BitcoinHeistData_complete.csv",
]
root_dir = "data/MNAR1Data/"
mnar_miss_rates = [0.05, 0.10, 0.20]
set_or_agg =[False]
# numeric_only_mnar=True ## for agg
# choose seeds to freeze splits + MAR/MCAR masks (optional)
SPLIT_SEED = 999
MASK_SEED  = 777   # same across runs -> MAR/MCAR masks identical across 5/10/20

for numeric_only_mnar in set_or_agg:
    for complete_dataset in complete_datasets_names:
        filename = os.path.basename(complete_dataset)                 #  "bank_complete.csv"
        name_without_ext = os.path.splitext(filename)[0]              # "bank_complete"
        first_word = name_without_ext.split("_")[0]
        # if numeric_only_mnar:
        #       first_word= first_word+"_agg_"                # "bank"
        folder_path = os.path.join(root_dir, first_word)
        if numeric_only_mnar:
            first_word= first_word+"_agg"
        os.makedirs(folder_path, exist_ok=True)
        out_csv=folder_path+f"/{first_word}_mnar1_injected.csv"
        print("main out csv : ----", out_csv)
        out_json=folder_path+f"/{first_word}_mnar1_dag.json"
        print("main out json : ----", out_json)
        out_png_regular=folder_path+f"/{first_word}_mnar1_mgraph.png"
        # Build an injector just to DISCOVER & PLAN once
        plan_builder = MnarInjector(
            in_path=complete_dataset,                                  # <-- use the current dataset
            out_csv=out_csv, out_json=out_json, out_png_regular=out_png_regular, seed=1337
        )
        plan = plan_builder.plan_injection(
            alpha=0.05, bins_per_numeric=5, split_seed=SPLIT_SEED, mcar_rate=0.15,numeric_only_mnar=numeric_only_mnar
        )
        if "note" in plan:
            print(first_word, plan["note"])
            continue
        print("----first_word-----",first_word)
        # Now run the SAME plan for each MNAR rate
        for mnar_rate in mnar_miss_rates:
            mrate = f"{int(mnar_rate*100)}"                        # 5, 10, 20
            out_csv  = os.path.join(folder_path, f"{first_word}_mnar1_{mrate}.csv")
            out_json = os.path.join(folder_path, f"{first_word}_mnar1_dag_{mrate}.json")
            out_png  = os.path.join(folder_path, f"{first_word}_mnar1_m_graph{mrate}.png")
            # out_png  = os.path.join(folder_path, f"{first_word}_mnar1_{rate_tag}.png")

            inj = MnarInjector(
                in_path=complete_dataset,
                out_csv=out_csv,
                out_json=out_json,
                out_png_regular=out_png,
                seed=1337                                             # base seed (structure)
            )

            inj.run_plan(
                plan,
                rate_mnar=mnar_rate,
                mar_rate=0.25,
                beta_mar=0.8,
                beta_prev=1.0,
                rng_seed=MASK_SEED                                    # freeze MAR/MCAR masks across rates
            )
            inj.save_outputs()
            inj.plot_m_graph()
            print(f"[OK] {first_word}: MNAR={mnar_rate:.2%} -> {out_csv}")

        discovered_complete_json = folder_path+f"/{first_word}_mnar1_discovered_dag.json"
        mechamic_json = folder_path+f"/{first_word}_mnar1_dag_5.json"
        seprators_out_csv =folder_path+f"/{first_word}_mnar1_minimal_separators_Xi.csv"

        find_serparators_inJson = minimalSeparatorsSearch(discovered_complete_json,mechamic_json,seprators_out_csv)

        find_serparators_inJson.findMySeparators()
