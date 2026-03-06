"""
SAP Knowledge Graph — SAP ECC Domain Knowledge Graph
=====================================================

A NetworkX-based knowledge graph that models the SAP ECC schema as an
interconnected graph of modules, tables, business concepts, and relationships.

Used for:
1. Resolving natural language business terms to schema elements
2. Discovering join paths between tables (shortest path traversal)
3. Providing explainability for query generation
4. Powering a visual graph explorer in the UI

Architecture:
  - Node types: MODULE, TABLE, CONCEPT, NL_TERM
  - Edge types: BELONGS_TO, FOREIGN_KEY, MEASURES/DESCRIBES, SYNONYM, RELATES_TO
  - Graph is built from the SAP semantic model JSON
  - Serializes to/from JSON for persistence
  - Exposes methods for the UI graph explorer
"""

import networkx as nx
import json
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path

_log = logging.getLogger("sap_kg")

# ───────────────────────────────────────────────────────────────────────
# Node types
# ───────────────────────────────────────────────────────────────────────

NODE_MODULE = "MODULE"
NODE_TABLE = "TABLE"
NODE_CONCEPT = "CONCEPT"
NODE_NL_TERM = "NL_TERM"

# ───────────────────────────────────────────────────────────────────────
# Edge types
# ───────────────────────────────────────────────────────────────────────

EDGE_BELONGS_TO = "BELONGS_TO"        # table belongs to module
EDGE_FK = "FOREIGN_KEY"                # table to table FK
EDGE_MEASURES = "MEASURES"             # concept to table
EDGE_DESCRIBES = "DESCRIBES"           # concept to table
EDGE_SYNONYM = "SYNONYM"               # NL term to concept/table
EDGE_RELATES_TO = "RELATES_TO"         # cross-module relationship

# ───────────────────────────────────────────────────────────────────────
# Module color scheme (SAP)
# ───────────────────────────────────────────────────────────────────────

MODULE_COLORS = {
    "FI_GL": "#6366f1",   # indigo
    "FI_AP": "#ec4899",   # pink
    "FI_AR": "#f59e0b",   # amber
    "CO": "#10b981",      # emerald
    "MM": "#3b82f6",      # blue
    "SD": "#8b5cf6",      # violet
    "PM": "#ef4444",      # red
    "HR": "#14b8a6",      # teal
    "PAY": "#f97316",     # orange
    "BEN": "#06b6d4",     # cyan
}

MODULE_NAMES = {
    "FI_GL": "General Ledger",
    "FI_AP": "Accounts Payable",
    "FI_AR": "Accounts Receivable",
    "CO": "Controlling",
    "MM": "Materials Management",
    "SD": "Sales & Distribution",
    "PM": "Plant Maintenance",
    "HR": "Human Resources",
    "PAY": "Payroll",
    "BEN": "Benefits",
}


class KnowledgeGraph:
    """
    A knowledge graph for SAP ECC schema, relationships, and concepts.
    """

    def __init__(self, model_path: str):
        """
        Initialize and build knowledge graph from SAP semantic model.

        Args:
            model_path: Path to sap_semantic_model.json
        """
        self.model_path = Path(model_path)
        self.graph = nx.MultiDiGraph()
        self._stats = {}
        self._nl_index = defaultdict(list)  # term -> [(node_id, node_type), ...]

        # Load and build
        self._load_model()
        self._build_graph()
        self._update_stats()

    def _load_model(self):
        """Load SAP semantic model from JSON."""
        with open(self.model_path, "r") as f:
            self.model = json.load(f)
        _log.info(f"Loaded SAP semantic model from {self.model_path}")

    def _build_graph(self):
        """Build the complete knowledge graph."""
        _log.info("Building SAP knowledge graph...")
        self._add_modules()
        self._add_tables()
        self._add_business_concepts()
        self._add_cross_module_relationships()
        self._build_nl_index()
        _log.info("Knowledge graph built successfully")

    def _add_modules(self):
        """Add module nodes for each SAP module."""
        modules = self.model.get("modules", {})

        for module_code, module_data in modules.items():
            module_name = MODULE_NAMES.get(module_code, module_data.get("module_name", module_code))
            color = MODULE_COLORS.get(module_code, "#9ca3af")
            description = module_data.get("description", "")

            node_id = f"mod:{module_code}"
            self.graph.add_node(
                node_id,
                node_type=NODE_MODULE,
                code=module_code,
                name=module_name,
                color=color,
                description=description,
                label=f"{module_code} — {module_name}",
            )

    def _add_tables(self):
        """Add table nodes and populate with table information."""
        modules = self.model.get("modules", {})
        added_tables = set()  # Track to avoid duplicates

        for module_code, module_data in modules.items():
            business_objects = module_data.get("business_objects", {})

            for concept_name, concept_data in business_objects.items():
                # Tables are stored as a dictionary under "tables" key
                tables_dict = concept_data.get("tables", {})

                for table_name, table_info in tables_dict.items():
                    if not table_name:
                        continue

                    # Skip if already added
                    if table_name in added_tables:
                        continue
                    added_tables.add(table_name)

                    # Extract description from table info
                    if isinstance(table_info, dict):
                        table_desc = table_info.get("description", "")
                    else:
                        table_desc = ""

                    node_id = f"tbl:{table_name}"
                    color = MODULE_COLORS.get(module_code, "#9ca3af")

                    self.graph.add_node(
                        node_id,
                        node_type=NODE_TABLE,
                        table_name=table_name,
                        module=module_code,
                        color=color,
                        description=table_desc,
                        label=table_name,
                    )

                    # Add BELONGS_TO edge
                    module_node = f"mod:{module_code}"
                    self.graph.add_edge(
                        node_id,
                        module_node,
                        edge_type=EDGE_BELONGS_TO,
                        label="belongs to",
                    )

    def _add_business_concepts(self):
        """
        Add business concept nodes and link them to tables.
        Extract concepts from business_objects and nl_aliases.
        """
        modules = self.model.get("modules", {})

        for module_code, module_data in modules.items():
            business_objects = module_data.get("business_objects", {})

            for concept_name, concept_data in business_objects.items():
                node_id = f"concept:{module_code}_{concept_name}"
                color = MODULE_COLORS.get(module_code, "#9ca3af")
                description = concept_data.get("description", "")
                nl_aliases = concept_data.get("nl_aliases", [])

                # Tables are stored in a dictionary
                tables_dict = concept_data.get("tables", {})

                self.graph.add_node(
                    node_id,
                    node_type=NODE_CONCEPT,
                    concept_name=concept_name,
                    module=module_code,
                    color=color,
                    description=description,
                    label=concept_name.replace("_", " ").title(),
                )

                # Link concept to tables it describes
                for table_name in tables_dict.keys():
                    if table_name:
                        table_node = f"tbl:{table_name}"
                        self.graph.add_edge(
                            node_id,
                            table_node,
                            edge_type=EDGE_DESCRIBES,
                            label="describes",
                        )

                # Add NL aliases as synonym edges
                for alias in nl_aliases:
                    nl_term_id = f"nlterm:{alias.lower()}"

                    # Create NL term node if it doesn't exist
                    if nl_term_id not in self.graph:
                        self.graph.add_node(
                            nl_term_id,
                            node_type=NODE_NL_TERM,
                            term=alias,
                            label=alias,
                            description=f"Natural language term for {concept_name}",
                        )

                    # Link NL term to concept
                    self.graph.add_edge(
                        nl_term_id,
                        node_id,
                        edge_type=EDGE_SYNONYM,
                        label="refers to",
                    )

                    # Index for quick lookup
                    self._nl_index[alias.lower()].append((node_id, NODE_CONCEPT))

    def _add_cross_module_relationships(self):
        """
        Add foreign key relationships between tables across modules.
        These come from the cross_module_relationships section.
        """
        relationships = self.model.get("cross_module_relationships", [])

        for rel in relationships:
            from_table = rel.get("from_table", "")
            to_table = rel.get("to_table", "")
            description = rel.get("description", "")
            join_condition = rel.get("join_condition", "")

            if not from_table or not to_table:
                continue

            from_node = f"tbl:{from_table}"
            to_node = f"tbl:{to_table}"

            # Only add if both tables exist in graph
            if from_node in self.graph and to_node in self.graph:
                self.graph.add_edge(
                    from_node,
                    to_node,
                    edge_type=EDGE_FK,
                    label="joins to",
                    description=description,
                    join_condition=join_condition,
                )

    def _build_nl_index(self):
        """
        Build reverse index from natural language terms to graph nodes.
        Also index concept names and table names.
        """
        # Index concept names
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == NODE_CONCEPT:
                concept_name = data.get("concept_name", "").lower()
                if concept_name:
                    self._nl_index[concept_name].append((node_id, NODE_CONCEPT))

            # Index table names
            if data.get("node_type") == NODE_TABLE:
                table_name = data.get("table_name", "").lower()
                if table_name:
                    self._nl_index[table_name].append((node_id, NODE_TABLE))

    def _update_stats(self):
        """Calculate and cache graph statistics."""
        self._stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "modules": sum(
                1 for _, data in self.graph.nodes(data=True) if data.get("node_type") == NODE_MODULE
            ),
            "tables": sum(
                1 for _, data in self.graph.nodes(data=True) if data.get("node_type") == NODE_TABLE
            ),
            "concepts": sum(
                1 for _, data in self.graph.nodes(data=True) if data.get("node_type") == NODE_CONCEPT
            ),
            "nl_terms": sum(
                1 for _, data in self.graph.nodes(data=True) if data.get("node_type") == NODE_NL_TERM
            ),
        }

    @property
    def stats(self) -> Dict:
        """Return graph statistics."""
        return self._stats.copy()

    def to_d3_json(self) -> Dict:
        """
        Export graph in D3.js force-directed format for visualization.

        Returns:
            Dict with 'nodes', 'links', and 'stats' keys for D3 visualization.
        """
        nodes = []
        links = []
        node_ids = set()

        # Add module, table, and concept nodes
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "")
            if node_type in (NODE_MODULE, NODE_TABLE, NODE_CONCEPT):
                node = {
                    "id": node_id,
                    "label": data.get("label", node_id),
                    "type": node_type,
                    "module": data.get("module", data.get("code", "")),
                    "description": data.get("description", "")[:100],
                    "group": data.get("module", data.get("code", "MISC")),
                    "color": data.get("color", "#9ca3af"),
                }
                nodes.append(node)
                node_ids.add(node_id)

        # Add edges between nodes we included
        for from_node, to_node, data in self.graph.edges(data=True):
            if from_node in node_ids and to_node in node_ids:
                link = {
                    "source": from_node,
                    "target": to_node,
                    "type": data.get("edge_type", "RELATES_TO"),
                    "label": data.get("label", ""),
                }
                links.append(link)

        return {
            "nodes": nodes,
            "links": links,
            "stats": self.stats,
        }

    def get_concept_schema(self, concept_name: str) -> Optional[Dict]:
        """
        Get detailed schema for a business concept.

        Args:
            concept_name: Name of the business concept

        Returns:
            Dict with concept details and related tables, or None if not found
        """
        # Search for the concept node
        concept_node = None
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") == NODE_CONCEPT:
                if data.get("concept_name", "").lower() == concept_name.lower():
                    concept_node = node_id
                    break

        if not concept_node:
            return None

        data = self.graph.nodes[concept_node]
        tables = []

        # Find all tables this concept describes
        for successor in self.graph.successors(concept_node):
            if self.graph.nodes[successor].get("node_type") == NODE_TABLE:
                table_data = self.graph.nodes[successor]
                tables.append({
                    "table_name": table_data.get("table_name", ""),
                    "module": table_data.get("module", ""),
                    "description": table_data.get("description", ""),
                })

        return {
            "concept_name": data.get("concept_name", ""),
            "module": data.get("module", ""),
            "description": data.get("description", ""),
            "tables": tables,
        }

    def get_table_context(self, table_name: str) -> Optional[Dict]:
        """
        Get context for a table: module, related concepts, and join partners.

        Args:
            table_name: Name of the table

        Returns:
            Dict with table context, or None if not found
        """
        table_node = f"tbl:{table_name}"
        if table_node not in self.graph:
            return None

        data = self.graph.nodes[table_node]
        module = data.get("module", "")
        module_node = f"mod:{module}"

        # Find related concepts
        concepts = []
        for predecessor in self.graph.predecessors(table_node):
            pred_data = self.graph.nodes[predecessor]
            if pred_data.get("node_type") == NODE_CONCEPT:
                concepts.append({
                    "concept_name": pred_data.get("concept_name", ""),
                    "description": pred_data.get("description", ""),
                })

        # Find join partners
        join_partners = []
        for successor in self.graph.successors(table_node):
            succ_data = self.graph.nodes[successor]
            if succ_data.get("node_type") == NODE_TABLE:
                join_partners.append({
                    "table_name": succ_data.get("table_name", ""),
                    "module": succ_data.get("module", ""),
                })

        return {
            "table_name": table_name,
            "module": module,
            "description": data.get("description", ""),
            "module_name": MODULE_NAMES.get(module, module),
            "concepts": concepts,
            "join_partners": join_partners,
        }

    def get_module_graph(self, module_code: str) -> Dict:
        """
        Get subgraph for a specific module (tables and concepts only).

        Args:
            module_code: Module code (e.g., "FI_GL", "SD")

        Returns:
            Dict with nodes and links for that module
        """
        nodes = []
        links = []
        node_ids = set()

        # Collect all nodes in this module
        for node_id, data in self.graph.nodes(data=True):
            module = data.get("module", "")
            node_type = data.get("node_type", "")

            if module == module_code and node_type in (NODE_TABLE, NODE_CONCEPT):
                node = {
                    "id": node_id,
                    "label": data.get("label", node_id),
                    "type": node_type,
                    "description": data.get("description", "")[:100],
                }
                nodes.append(node)
                node_ids.add(node_id)

        # Collect edges between nodes in this module
        for from_node, to_node, data in self.graph.edges(data=True):
            if from_node in node_ids and to_node in node_ids:
                link = {
                    "source": from_node,
                    "target": to_node,
                    "type": data.get("edge_type", ""),
                    "label": data.get("label", ""),
                }
                links.append(link)

        return {
            "module": module_code,
            "module_name": MODULE_NAMES.get(module_code, module_code),
            "nodes": nodes,
            "links": links,
        }

    def resolve_nl_term(self, term: str) -> List[Tuple[str, str, Dict]]:
        """
        Resolve a natural language term to graph nodes.

        Args:
            term: Natural language term to resolve

        Returns:
            List of (node_id, node_type, node_data) tuples
        """
        results = []
        term_lower = term.lower()

        # Direct lookup in index
        for node_id, node_type in self._nl_index.get(term_lower, []):
            if node_id in self.graph:
                results.append((node_id, node_type, dict(self.graph.nodes[node_id])))

        # Fuzzy matching: check for substrings
        if not results:
            for indexed_term, nodes in self._nl_index.items():
                if term_lower in indexed_term or indexed_term in term_lower:
                    for node_id, node_type in nodes:
                        if node_id in self.graph:
                            results.append((node_id, node_type, dict(self.graph.nodes[node_id])))

        return results

    def resolve_question(self, question: str) -> Dict:
        """
        Resolve a question to relevant tables and concepts.

        Args:
            question: Natural language question

        Returns:
            Dict with relevant tables, concepts, and suggested joins
        """
        # Tokenize and extract key terms
        terms = question.lower().split()
        tables = set()
        concepts = set()

        for term in terms:
            # Remove common words
            if term in ("what", "show", "me", "the", "is", "are", "for", "of", "a", "an", "by", "and"):
                continue

            # Try to resolve the term
            results = self.resolve_nl_term(term)
            for node_id, node_type, data in results:
                if node_type == NODE_TABLE:
                    tables.add(node_id)
                elif node_type == NODE_CONCEPT:
                    concepts.add(node_id)
                    # Also get tables for this concept
                    for successor in self.graph.successors(node_id):
                        if self.graph.nodes[successor].get("node_type") == NODE_TABLE:
                            tables.add(successor)

        # Find join paths between tables
        join_paths = []
        table_list = list(tables)
        for i in range(len(table_list)):
            for j in range(i + 1, len(table_list)):
                paths = self.find_all_join_paths(table_list[i], table_list[j])
                join_paths.extend(paths[:1])  # Keep first path

        return {
            "question": question,
            "tables": [{"id": t, "data": dict(self.graph.nodes[t])} for t in tables],
            "concepts": [{"id": c, "data": dict(self.graph.nodes[c])} for c in concepts],
            "suggested_joins": join_paths,
        }

    def find_join_path(self, table1: str, table2: str) -> Optional[List[str]]:
        """
        Find shortest join path between two tables.

        Args:
            table1: Source table (e.g., "BSEG" or "tbl:BSEG")
            table2: Target table (e.g., "BSIK" or "tbl:BSIK")

        Returns:
            List of node IDs representing shortest path, or None if no path
        """
        # Normalize table names
        node1 = table1 if table1.startswith("tbl:") else f"tbl:{table1}"
        node2 = table2 if table2.startswith("tbl:") else f"tbl:{table2}"

        if node1 not in self.graph or node2 not in self.graph:
            return None

        try:
            return nx.shortest_path(self.graph, node1, node2)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return None

    def find_all_join_paths(self, table1: str, table2: str, max_paths: int = 5) -> List[List[str]]:
        """
        Find all join paths between two tables (limited by max_paths).

        Args:
            table1: Source table
            table2: Target table
            max_paths: Maximum number of paths to return

        Returns:
            List of paths, each path is a list of node IDs
        """
        # Normalize table names
        node1 = table1 if table1.startswith("tbl:") else f"tbl:{table1}"
        node2 = table2 if table2.startswith("tbl:") else f"tbl:{table2}"

        if node1 not in self.graph or node2 not in self.graph:
            return []

        try:
            all_paths = nx.all_simple_paths(self.graph, node1, node2, cutoff=4)
            return list(all_paths)[:max_paths]
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []


def main():
    """Build and save the SAP knowledge graph."""
    model_path = Path(__file__).parent / "sap_semantic_model.json"
    output_path = Path(__file__).parent / "sap_knowledge_graph.json"

    kg = KnowledgeGraph(str(model_path))

    # Print stats
    print("\n" + "=" * 70)
    print("SAP Knowledge Graph Statistics")
    print("=" * 70)
    for key, value in kg.stats.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # Export to D3 JSON
    d3_json = kg.to_d3_json()
    with open(output_path, "w") as f:
        json.dump(d3_json, f, indent=2)

    print(f"\nKnowledge graph exported to {output_path}")
    print(f"  Nodes: {len(d3_json['nodes'])}")
    print(f"  Links: {len(d3_json['links'])}")


if __name__ == "__main__":
    main()
