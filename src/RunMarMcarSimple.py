#!/usr/bin/env python3
"""Simplified MAR/MCAR runner: direct-5.2 + ranking (with timeout). Writes results incrementally."""
import os, json, re, time, signal
import numpy as np
import pandas as pd
import psycopg2
from RankingQueryExecuter import QueryExecutorRanking as RankExecutor
from RunnerSetQueriy import to_executor_csv_queries, build_maps_from_lists, replace_csv_with_tables
from nonAgg_direct import run_direct_per_tuple, _load_table

CONN = dict(host="localhost", port=5433, dbname="mydb", user="alzamill", password=os.environ.get("PGPASSWORD", ""))
OUT = "mar_mcar_set_results.csv"
RANK_TIMEOUT = 60

def nv(v):
    s = str(v)
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else s
    except: return s

def nt(r): return tuple(nv(x) for x in r)

def tv_sets(ps, gs):
    if not ps and not gs: return 0.0
    p = 1.0/len(ps) if ps else 0.0
    q = 1.0/len(gs) if gs else 0.0
    return 0.5*sum(abs((p if t in ps else 0)-(q if t in gs else 0)) for t in ps|gs)

class Timeout(Exception): pass
def alarm_handler(s, f): raise Timeout()

def main():
    # Clean DB
    c = psycopg2.connect(**CONN); c.autocommit=True
    c.cursor().execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename='alzamill' AND pid<>pg_backend_pid()")
    c.close()

    conn = psycopg2.connect(**CONN); conn.autocommit=False
    rows = []

    for mt, jp in [("MAR","configs/mar_set_queries.json"),("MCAR","configs/mcar_set_queries.json")]:
        with open(jp) as f: cfg = json.load(f)
        for gk in cfg:
            for bk, meta in cfg[gk].items():
                csvs=meta["csv"]; tables=meta["table"]
                ccsvs=meta.get("complete_csv",[]); ctabs=meta.get("complete_table",[])
                ma=[a.lower() for a in meta.get("missing_attrs_single",[])]
                od={k.lower():[c.lower() for c in v] for k,v in meta.get("ordering_single",{}).items()}
                qs=[q for q in meta.get("queries",[]) if "JOIN" not in q.upper()]
                if not csvs or not qs: continue

                print("\n%s / %s" % (mt, bk), flush=True)
                # Reconnect if needed
                try: conn.rollback(); conn.cursor().execute("SELECT 1")
                except:
                    try: conn.close()
                    except: pass
                    conn = psycopg2.connect(**CONN); conn.autocommit=False

                try:
                    _load_table(conn, csvs[0], tables[0], force=True)
                    if ccsvs and ctabs: _load_table(conn, ccsvs[0], ctabs[0])
                except Exception as e:
                    print("  LOAD ERR: %s" % str(e)[:50], flush=True)
                    try: conn.rollback()
                    except: pass
                    continue

                cq = to_executor_csv_queries(bk, csvs, tables, ccsvs, ctabs)
                fm, bm = build_maps_from_lists(csvs, tables)
                # GT map
                gtm={}; gtb={}
                for cp,gt in zip(csvs, ctabs or tables):
                    gtm[cp]=gt; bn=os.path.basename(cp); gtb[bn]=gt
                    s=os.path.splitext(bn)[0]; d=os.path.basename(os.path.dirname(cp)) or "."
                    gtb["%s/%s"%(d,bn)]=gt; gtb["%s/%s"%(d,s)]=gt

                for qi,q in enumerate(qs):
                    psql = replace_csv_with_tables(q, fm, bm)
                    gsql = replace_csv_with_tables(q, gtm, gtb)
                    # GT
                    gt=set()
                    try:
                        conn.rollback(); cur=conn.cursor(); cur.execute(gsql)
                        gt={nt(r) for r in cur.fetchall()}; cur.close(); conn.commit()
                    except:
                        try: conn.rollback()
                        except: pass

                    rb = {"miss_type":mt,"block":bk,"query_idx":qi+1}

                    # Ranking
                    print("  Q%d rank.."%(qi+1), end="", flush=True)
                    try:
                        conn.rollback()
                        re2 = RankExecutor(conn, cq, skip_prepare=True)
                        re2._ordering_T=od; re2._missing_T=ma; re2.FRACTION=0.5
                        old=signal.signal(signal.SIGALRM, alarm_handler)
                        signal.alarm(RANK_TIMEOUT)
                        try:
                            t0=time.time()
                            fr,iw,tp,hb=re2.run_flat(psql,od,ma,None,None)
                            el=time.time()-t0; signal.alarm(0)
                        except Timeout:
                            signal.alarm(0); raise Exception("timeout")
                        finally: signal.signal(signal.SIGALRM,old)
                        tn=4 if hb else 2
                        ps2={}
                        for r in fr:
                            pay=nt(r[:-tn]); sc=max(0,float(r[-tn] or 0))
                            if pay not in ps2 or sc>ps2[pay]: ps2[pay]=sc
                        zp=sum(ps2.values()); zg=max(len(gt),1)
                        tv=0.0
                        for t in set(ps2)|gt:
                            tv+=abs((ps2.get(t,0)/zp if zp>1e-12 else 0)-((1.0/zg) if t in gt else 0))
                        tv*=0.5
                        print(" %.4f"%tv, flush=True)
                    except Exception as e:
                        tv=float("nan"); el=0
                        print(" ERR:%s"%str(e)[:30], flush=True)
                        try: conn.rollback()
                        except: pass
                    rows.append({**rb,"method":"ranking","tv_prob":tv,"time_s":el})

                    # Direct 5.2
                    try: conn.rollback()
                    except: pass
                    print("  Q%d dir.."%( qi+1), end="", flush=True)
                    try:
                        r5=run_direct_per_tuple(conn,q,tables[0],ctabs[0] if ctabs else tables[0],ma,od)
                        if r5.get("error"): tv5=float("nan"); print(" ERR",flush=True)
                        else: tv5=r5["tv_prob"]; print(" %.4f dw=%.4f"%(tv5,r5.get("delta_w",0)),flush=True)
                    except Exception as e:
                        tv5=float("nan"); print(" ERR:%s"%str(e)[:30],flush=True)
                        try: conn.rollback()
                        except: pass
                    rows.append({**rb,"method":"direct-5.2","tv_prob":tv5,"time_s":r5.get("sql_time_s",0) if not np.isnan(tv5) else 0})

                    # Save incrementally
                    pd.DataFrame(rows).to_csv(OUT, index=False)

    conn.close()
    df=pd.DataFrame(rows)
    df.to_csv(OUT,index=False)
    print("\nSaved %s (%d rows)"%(OUT,len(df)),flush=True)
    g=df.dropna(subset=["tv_prob"])
    if len(g)>0:
        print(g.groupby(["miss_type","block","method"])["tv_prob"].mean().to_string())

if __name__=="__main__": main()
