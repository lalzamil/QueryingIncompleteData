# Builds minimal separating sets X_i for each Y using the two JSON files from the DAGdisoveredovery.py


import json
import itertools
from collections import defaultdict, deque
import pandas as pd
import numpy as np


class minimalSeparatorsSearch:
    def __init__(self,disovered_json,mechanisms_json, out_separtors_json):

        self.out_separtors_json=out_separtors_json
        with open(disovered_json, "r") as f:
            self.disovered = json.load(f)

        with open(mechanisms_json, "r") as f:
            self.mechanisms = json.load(f)  # has "self.mechanisms": {"R_y": {"type":..., "causes":[...]}, ...}

        self.parents_map = self.disovered["parents"]                     # dict: var -> list of parent vars  (data DAG)
        self.topo_order  = self.disovered.get("topo_order", list(self.parents_map.keys()))

        self.mechanisms  = self.mechanisms["mechanisms"]                # dict: "R_y" -> {"causes":[...], "type":...}

        self.data_attr = list(self.parents_map.keys())
        self.R_nodes   = list(self.mechanisms.keys())

        # build augmented dag G (m-graph
        # mgraph G has edges (data_parent -> data_child)   from disovered json
        #              (cause -> R_y)                from mechanisms json
        self.G = defaultdict(set)

        for child, pas in self.parents_map.items():
            for p in pas:
                if p != child:
                    self.G[p].add(child)

        for Ry, meta in self.mechanisms.items():
            for c in meta.get("causes", []):
                if c != Ry:
                    self.G[c].add(Ry)

        self.G_rev = self.reverse_graph(self.G) #to compute ancestors
        pass



    def reverse_graph(self,G): #to compute ancestors
        R = defaultdict(set)
        for u, outs in G.items():
            for v in outs:
                R[v].add(u)
        return R



    def ancestors(self,G_rev, nodes):
        #Returns the ancestrors set of a node set
        vis = set(nodes)
        Q= deque(nodes)
        while Q:
            v = Q.popleft()
            for p in G_rev.get(v, ()):
                if p not in vis:
                    vis.add(p); Q.append(p)
        return vis

    def moralize(self,G, U):
        """
        Replace every directed edge by an undirected edge
        and connects  nodes that share a child
          """
        undirect = defaultdict(set)
        # add undirected versions of directed edges within U
        for u, outs in G.items():
            if u not in U:
                continue
            for v in outs:
                if v in U:
                    undirect[u].add(v)
                    undirect[v].add(u)
        # connect co-parents
        rev = defaultdict(set)
        for u, outs in G.items():
            for v in outs:
                if u in U and v in U:
                    rev[v].add(u)
        for v, pas in rev.items():
            pasU = [p for p in pas if p in U]
            for a, b in itertools.combinations(pasU, 2):
                undirect[a].add(b); undirect[b].add(a)
        return undirect

    def d_separated(self,G, G_rev, A, B, C):
        """
        D-separation test Lauritzen–Spirtes, Pearl for disjoint node sets A, B given conditioning set C.
        1) Find ancestral set of A∪B∪C via reverse graph.
        2) Moralize the subgraph.
        3) Remove C (and incident edges).
        4) Check connectivity between any a in A and b in B via BFS.
        """
        A, B, C = set(A), set(B), set(C)
        U = self.ancestors(G_rev, A | B | C)
        undirect = self.moralize(G, U)
        # remove C
        for c in C:
            for nb in list(undirect.get(c, ())):
                undirect[nb].discard(c)
            undirect.pop(c, None)
        # BFS from A # if we can reacha n node in B, the A amd B are not d-sep by C, else they are
        visited = set(A)
        dq = deque(list(A))
        while dq:
            x = dq.popleft()
            for y in undirect.get(x, ()):
                if y in C or y in visited:
                    continue
                if y in B:
                    return False
                visited.add(y); dq.append(y)
        return True

    #  minimal separators search
    def minimal_separator_for_Y(self,G, G_rev, parents_map, mechanisms, Y):
        """
        Find a minimal Xi ⊆ data_attr such that
        (i) Y ⟂ R_Y | Xi
        (ii) Y ⟂ R_X | Xi  for all X ∈ Xi

        - start with S = causes(R_Y) (cuases from the injector )
        - if needed, augment with structural parents/ancestors of Y
        - greedily prune to minimality
        """
        Ry = f"R_{Y}"
        if Ry not in mechanisms:
            return []  # fully observed in mechanisms json

        # Start from listed causes of R_Y intersected with substantive vars
        S = set(mechanisms[Ry].get("causes", [])) & set(self.data_attr)

        def constraints_hold(Sset):
            # (i) Y ⟂ R_Y | S
            if not self.d_separated(G, G_rev, {Y}, {Ry}, set(Sset)):
                return False
            # (ii) for all X in S: Y ⟂ R_X | S
            for X in list(Sset):
                RX = f"R_{X}"
                if RX in mechanisms:
                    if not self.d_separated(G, G_rev, {Y}, {RX}, set(Sset)):
                        return False
            return True

        # If initial S doesn't work (should rarely happen), augment with structural parents of Y
        if not constraints_hold(S):
            S |= set(parents_map.get(Y, []))
            # broaden to ancestors if still needed
            if not constraints_hold(S):
                ancY = (self.ancestors(G_rev, {Y}) - {Y}) & set(self.data_attr)
                S |= ancY

        # Greedy prune to minimality
        changed = True
        while changed:
            changed = False
            for x in list(S):
                trial = S - {x}
                if constraints_hold(trial):
                    S = trial
                    changed = True
        return sorted(S)

    def findMySeparators(self,):
        rows = []
        for Y in self.data_attr:
            Xi = self.minimal_separator_for_Y(self.G, self.G_rev, self.parents_map, self.mechanisms, Y)
            Ry = f"R_{Y}"
            mech = self.mechanisms.get(Ry, {"type": "FullyObserved", "causes": []})
            rows.append({
                "Y": Y,
                "type(R_Y)": mech.get("type", "FullyObserved"),
                "causes(R_Y)": mech.get("causes", []),
                "Xi_minimal": Xi,
                "|Xi|": len(Xi)
            })

        df_out = pd.DataFrame(rows).sort_values(["|Xi|","Y"]).reset_index(drop=True)
        print(df_out)
        df_out.to_csv(self.out_separtors_json, index=False) ####



# discovered_complete_json = "data/z_dagTests/bank_discovered_dag.json"
# mechamic_json = "z_dagTests/bank_mnar1_dag.json"
# seprators_out_csv ="z_dagTests/ban_mnar1_minimal_separators_Xi.csv"

# find_serparators_inJson = minimalSeparatorsSearch(discovered_complete_json,mechamic_json,seprators_out_csv)

# find_serparators_inJson.findMySeparators()
