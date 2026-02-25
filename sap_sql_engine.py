#!/usr/bin/env python3
"""
SAP ECC 6.0 Prompt-to-SQL Engine
===================================
Converts natural language business questions into SAP HANA SQL queries
using the SAP Semantic Object Model as context.

QUICK START:
    python3 sap_sql_engine.py --server
"""

import json
import os
import sys
import subprocess
import argparse
import textwrap
import webbrowser
import threading
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# ---------------------------------------------------------------------------
# Paths & Configuration
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(APP_DIR, "sap_semantic_model.json")
CONFIG_FILE = os.path.join(APP_DIR, "config.json")
HTML_FILE = os.path.join(APP_DIR, "sap_sql_ui.html")
TEST_DB_FILE = os.path.join(APP_DIR, "sap_test.db")
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096
DEFAULT_PORT = 8766


# ---------------------------------------------------------------------------
# Config management (persistent API key storage)
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """Load config from config.json, creating defaults if missing."""
    defaults = {
        "anthropic_api_key": "",
        "model": CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        "port": DEFAULT_PORT
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            defaults.update(saved)
        except (json.JSONDecodeError, IOError):
            pass
    return defaults


def save_config(config: dict):
    """Save config to config.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_api_key(config: dict) -> str:
    """Get API key from config file or environment variable."""
    return os.environ.get("ANTHROPIC_API_KEY", "") or config.get("anthropic_api_key", "")


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------
def ensure_anthropic_installed():
    """Install the anthropic package if it's not already available."""
    try:
        import anthropic
        return True
    except ImportError:
        print("Installing anthropic package...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "anthropic", "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print("  Done.")
            return True
        except subprocess.CalledProcessError:
            print("  Could not auto-install. Run: pip3 install anthropic")
            return False


# ---------------------------------------------------------------------------
# Load semantic model
# ---------------------------------------------------------------------------
def load_semantic_model(path: str = MODEL_FILE) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------
def build_system_prompt(model: dict) -> str:
    """
    Build the system prompt that teaches the LLM about the SAP ECC schema.
    Transforms the semantic model into effective LLM context for SQL generation.
    """
    parts = []

    parts.append(textwrap.dedent("""\
    You are an SAP ECC 6.0 SQL expert. Your role is to convert natural language
    business questions into accurate, production-ready SQL queries against SAP
    tables using the schema metadata provided below.

    RULES:
    1. Generate ANSI-compatible SQL (suitable for SAP HANA, but avoid HANA-only syntax)
    2. Use ONLY tables and columns defined in the schema below — never guess or invent columns
    3. Always join through defined foreign key relationships
    4. Exclude deleted records (LOEKZ = 'X') unless specifically asked
    5. Use COALESCE() for nullable numeric columns in calculations
    6. Use UPPER() for case-insensitive text matching
    7. SAP dates are stored as CHAR(8) in YYYYMMDD format — compare as strings
    8. DMBTR (amount in local currency) is always positive; use SHKZG to determine sign:
       - SHKZG = 'S' means Debit (positive)
       - SHKZG = 'H' means Credit (negative)
       - Net amount = CASE WHEN SHKZG = 'H' THEN -DMBTR ELSE DMBTR END
    9. Always filter by BUKRS (company code) when context is provided
    10. Include column aliases that match business terminology
    11. Add comments explaining the logic for complex queries
    12. Period amounts (HSL01-HSL16, WKG001-WKG012) correspond to fiscal periods 1-16
    13. SPRAS = 'E' for English text descriptions

    COLUMN VERIFICATION (MANDATORY — read this carefully):
    Rule 2 above is the most important rule. Before including ANY column in your SQL,
    verify it appears in the schema listings below for that table. Common mistakes to
    avoid:
    - Do NOT assume a table has columns just because standard SAP ECC would. This
      schema is a curated subset. If a column is not listed below, it does not exist.
    - If the user asks for data that requires columns not in the schema, do NOT generate
      SQL. Instead, respond conversationally: explain which columns ARE available on the
      relevant tables, and offer to write a query using those columns instead.
    - When in doubt, list only columns you can confirm from the schema below.

    PII PROTECTION RULES (MANDATORY — applies to HR, PAY, and BEN modules):
    Employee data is sensitive. You MUST enforce these rules for ANY query that touches
    PA0001, PA0002, PA0006, PA0008, PA0014, PA0167, PA0168, PA0169, HRPY_RGDIR,
    T511, or T512T:
    1. NEVER return individual employee rows. Do not generate SQL that would list or
       display individual employees by name, personnel number, date of birth, or any
       other personally identifiable attribute.
    2. NEVER expose individual salaries, pay details, or benefit enrollments tied to
       a specific person.
    3. ALWAYS aggregate. Queries against these tables must use GROUP BY at the
       organizational unit, personnel area, employee group, employee subgroup, pay
       scale, or cost center level. Every SELECT must use aggregate functions
       (COUNT, SUM, AVG, MIN, MAX) — never raw row-level detail.
    4. If the user explicitly asks for individual employee data, do NOT generate SQL.
       Instead, respond: "I'm unable to generate queries that return personally
       identifiable employee information. I can help with aggregate analyses instead —
       for example: (a) Average salary by organizational unit, (b) Headcount by
       personnel area, (c) Total payroll cost by cost center, (d) Benefit enrollment
       counts by plan type. Which would you like?"
    5. Even when filtering to a single org unit or cost center, the query must still
       aggregate — never return individual rows.

    CRITICAL — QUERY TEMPLATE RULE (highest priority):
    Before writing ANY SQL, scan the EXAMPLE QUERY PATTERNS section at the end of this
    prompt. If the user's question matches or closely resembles an example prompt, you
    MUST copy that example's SQL template VERBATIM as your query. Do not restructure,
    rewrite, simplify, or "improve" the template — only adjust literal filter values
    or parameters as needed. The templates have been tested and validated against the
    actual database. Generating your own SQL when a matching template exists will
    produce errors because the templates use specific subquery aliases and column
    expressions that are required for correctness.

    CLARIFICATION BEHAVIOR (IMPORTANT — follow these rules strictly):
    Before generating SQL, first evaluate whether the user's question is specific
    enough to produce a useful, unambiguous query. If ANY of these conditions apply,
    you MUST ask for clarification instead of generating SQL:

    1. VAGUE METRIC: The question says "show me sales" or "show me data" or similar
       without specifying what metric (revenue? count? volume? outstanding balance?)
    2. MISSING TIME PERIOD: The question says "recent" or "last" without a specific
       timeframe, or asks about trends without specifying a date range
    3. AMBIGUOUS SCOPE: The question could apply to multiple modules (e.g., "how much
       did we spend" could mean AP invoices, PO commitments, cost center postings, etc.)
    4. OUT OF SCOPE: The question references data not in the schema below
    5. TOO BROAD: The question would return an unmanageably large result set without
       filters (e.g., "show me all transactions")

    When asking for clarification:
    - Do NOT generate any SQL. Do NOT include a ```sql code block.
    - Be conversational and helpful
    - Offer 2-3 specific options labeled (a), (b), (c)
    - Example: "Could you clarify what you mean by 'expenses'? I can show you:
      (a) Cost center actual postings from CO (COSP/COEP)
      (b) Vendor invoice amounts from FI-AP (BSIK/BSAK)
      (c) Purchase order commitments from MM (EKKO/EKPO)"

    When NOT to ask for clarification — just generate the SQL:
    - The question clearly maps to a specific table/module
    - The question includes enough filters and specifics
    - A reasonable default interpretation exists (e.g., "list all vendors" is clear enough)

    RESPONSE FORMAT:
    - Start with a brief explanation of your approach (which objects/tables you'll use and why)
    - Then provide the SQL query in a ```sql code block
    - End with notes on any assumptions made or parameters the user should customize

    """))

    # Shared reference objects
    parts.append("\n=== SHARED REFERENCE OBJECTS ===\n")
    for obj_name, obj in model.get("shared_reference_objects", {}).items():
        parts.append(f"\n## {obj_name}")
        parts.append(f"Description: {obj['description']}")
        if obj.get("nl_aliases"):
            parts.append(f"NL Aliases: {', '.join(obj['nl_aliases'])}")
        for tbl_name, tbl in obj["tables"].items():
            parts.append(f"\n  Table: {tbl_name} ({tbl.get('schema', '')}.{tbl_name})")
            parts.append(f"  Description: {tbl['description']}")
            pk = tbl.get("primary_key", {})
            if pk:
                parts.append(f"  Primary Key: {pk.get('column', 'N/A')}")
            parts.append("  Columns:")
            for col in tbl["business_columns"]:
                aliases = f" (aka: {', '.join(col['nl_aliases'])})" if col.get("nl_aliases") else ""
                values = f" [values: {', '.join(col['common_values'])}]" if col.get("common_values") else ""
                parts.append(f"    - {col['column']} {col['type']}: {col['description']}{aliases}{values}")
            if tbl.get("usage_notes"):
                parts.append(f"  Usage: {tbl['usage_notes']}")

    # Module objects
    for mod_key, mod in model.get("modules", {}).items():
        parts.append(f"\n\n{'='*60}")
        parts.append(f"=== MODULE: {mod['module_name']} ({mod_key}) ===")
        parts.append(f"{'='*60}")
        parts.append(f"Description: {mod['description']}")

        for obj_name, obj in mod["business_objects"].items():
            parts.append(f"\n\n--- Business Object: {obj_name} ---")
            parts.append(f"Description: {obj['description']}")
            if obj.get("nl_aliases"):
                parts.append(f"NL Aliases: {', '.join(obj['nl_aliases'])}")
            if obj.get("business_questions"):
                parts.append(f"Common Questions: {'; '.join(obj['business_questions'][:3])}")

            for tbl_name, tbl in obj["tables"].items():
                parts.append(f"\n  Table: {tbl_name}")
                parts.append(f"  Description: {tbl['description']}")
                pk = tbl.get("primary_key", {})
                if pk:
                    parts.append(f"  Primary Key: {pk.get('column', 'N/A')}")
                fks = tbl.get("foreign_keys", [])
                if fks:
                    parts.append("  Foreign Keys:")
                    for fk in fks:
                        parts.append(f"    - {fk['column']} -> {fk['references']}")
                parts.append("  Columns:")
                for col in tbl["business_columns"]:
                    aliases = f" (aka: {', '.join(col['nl_aliases'])})" if col.get("nl_aliases") else ""
                    values = f" [values: {', '.join(col['common_values'])}]" if col.get("common_values") else ""
                    lookup = f" [lookup: {col['lookup_type']}]" if col.get("lookup_type") else ""
                    parts.append(f"    - {col['column']} {col['type']}: {col['description']}{aliases}{values}{lookup}")
                if tbl.get("usage_notes"):
                    parts.append(f"  Usage Note: {tbl['usage_notes']}")

            if obj.get("intra_object_relationships"):
                parts.append("\n  Internal Relationships:")
                for rel in obj["intra_object_relationships"]:
                    join_text = rel.get('join', rel.get('description', ''))
                    parts.append(f"    - {rel['from']} -> {rel['to']} ({rel['type']}): {join_text}")

    # Cross-module relationships
    parts.append("\n\n=== CROSS-MODULE RELATIONSHIPS ===\n")
    for rel in model.get("cross_module_relationships", []):
        # Support both OEBS format (name, from_object, join.from/to) and SAP format (from_table, to_table, join_condition)
        if "name" in rel:
            # OEBS format
            parts.append(f"  {rel['name']}:")
            parts.append(f"    {rel.get('from_object', '')} -> {rel.get('to_object', '')} ({rel.get('type', '')})")
            join_info = rel.get("join", {})
            if isinstance(join_info, dict):
                parts.append(f"    Join: {join_info.get('from', '')} = {join_info.get('to', '')}")
            elif isinstance(join_info, str):
                parts.append(f"    Join: {join_info}")
            if rel.get("nl_description"):
                parts.append(f"    NL: {rel['nl_description']}")
        else:
            # SAP format
            from_tbl = rel.get("from_table", "")
            to_tbl = rel.get("to_table", "")
            desc = rel.get("description", "")
            join_cond = rel.get("join_condition", "")
            parts.append(f"  {from_tbl} -> {to_tbl}:")
            parts.append(f"    Description: {desc}")
            parts.append(f"    Join: {join_cond}")
            from_mod = rel.get("from_module", "")
            to_mod = rel.get("to_module", "")
            if from_mod and to_mod:
                parts.append(f"    Modules: {from_mod} -> {to_mod}")

    # SQL generation guidelines
    guidelines = model.get("sql_generation_guidelines", {})
    if guidelines:
        parts.append("\n\n=== SQL GENERATION GUIDELINES ===\n")
        if isinstance(guidelines, list):
            for g in guidelines:
                if isinstance(g, dict):
                    parts.append(f"  {g.get('guideline', '')}: {g.get('description', '')}")
                    if g.get('example'):
                        parts.append(f"    Example: {g['example']}")
                else:
                    parts.append(f"  - {g}")
        elif isinstance(guidelines, dict):
            parts.append("General Rules:")
            for rule in guidelines.get("general_rules", []):
                parts.append(f"  - {rule}")
            if guidelines.get("common_term_mappings"):
                parts.append("\nCommon Term Mappings (NL term -> SQL expression):")
                for term, mapping in guidelines["common_term_mappings"].items():
                    parts.append(f"  \"{term}\" -> {mapping}")
            if guidelines.get("important_distinctions"):
                parts.append("\nImportant Distinctions:")
                for d in guidelines["important_distinctions"]:
                    if isinstance(d, dict):
                        parts.append(f"  {d['concept']}: {d['explanation']}")
                    else:
                        parts.append(f"  - {d}")
            if guidelines.get("multi_org_note"):
                parts.append(f"\nMulti-Org: {guidelines['multi_org_note']}")

    # Example query patterns (few-shot)
    patterns = model.get("nl_query_patterns", [])
    if patterns:
        parts.append("\n\n=== EXAMPLE QUERY PATTERNS (MANDATORY — use these templates verbatim) ===")
        parts.append("\nIf the user's question matches any example below, copy the SQL template EXACTLY.")
        parts.append("Do NOT rewrite it. These templates use tested subquery aliases and column expressions.\n")
        for p in patterns:
            # Support both OEBS format and SAP format
            if "example_prompt" in p:
                # OEBS format
                parts.append(f"\nExample: \"{p['example_prompt']}\"")
                parts.append(f"Objects: {', '.join(p.get('objects_used', []))}")
                parts.append(f"Tables: {', '.join(p.get('tables_used', []))}")
                parts.append(f"SQL:\n{p['sql_template']}")
                if p.get("key_concepts"):
                    kc = p["key_concepts"]
                    if isinstance(kc, dict):
                        for k, v in kc.items():
                            parts.append(f"  {k}: {v}")
                    elif isinstance(kc, list):
                        for item in kc:
                            parts.append(f"  - {item}")
            else:
                # SAP format
                parts.append(f"\nPattern: {p.get('pattern_name', '')}")
                parts.append(f"Description: {p.get('description', '')}")
                parts.append(f"Module: {p.get('primary_module', '')}")
                tables = p.get('tables', [])
                if tables:
                    parts.append(f"Tables: {', '.join(tables)}")
                parts.append(f"Context: {p.get('business_context', '')}")
                if p.get('example_sql_structure'):
                    parts.append(f"SQL Structure:\n{p['example_sql_structure']}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SQL generation via API
# ---------------------------------------------------------------------------
def generate_sql_with_api(question: str, system_prompt: str, api_key: str,
                          model_name: str = CLAUDE_MODEL,
                          conversation_history: list = None) -> str:
    """Generate SQL using the Anthropic API. Supports conversation history for follow-ups."""
    import anthropic
    try:
        import httpx
        http_client = httpx.Client(verify=False)
        client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
    except ImportError:
        client = anthropic.Anthropic(api_key=api_key)

    if conversation_history:
        messages = conversation_history + [{"role": "user", "content": question}]
    else:
        messages = [{"role": "user", "content": question}]

    response = client.messages.create(
        model=model_name,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages
    )
    return {
        "text": response.content[0].text,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    }


# ---------------------------------------------------------------------------
# SAP SQL to SQLite conversion
# ---------------------------------------------------------------------------
import sqlite3
import re as _re


def _find_balanced_args(s: str, start: int):
    """
    Given string s and position start pointing to the opening '(' of a function
    call, find the top-level comma-separated arguments respecting nested parens
    and quoted strings.  Returns (list_of_arg_strings, end_pos) where end_pos
    is the index of the closing ')'.  Returns None if unbalanced.
    """
    if start >= len(s) or s[start] != '(':
        return None
    depth = 1
    i = start + 1
    args = []
    arg_start = i
    while i < len(s) and depth > 0:
        ch = s[i]
        if ch == "'":
            # skip quoted string
            i += 1
            while i < len(s) and s[i] != "'":
                i += 1
        elif ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                args.append(s[arg_start:i].strip())
                return (args, i)
        elif ch == ',' and depth == 1:
            args.append(s[arg_start:i].strip())
            arg_start = i + 1
        i += 1
    return None


def _replace_function_calls(s: str, func_name: str, handler):
    """
    Find all occurrences of func_name(...) in s, parse balanced args,
    and call handler(args) -> replacement_string.  Processes right-to-left
    so replacements don't shift positions of earlier matches.
    """
    pattern = _re.compile(r'\b' + func_name + r'\s*\(', _re.IGNORECASE)
    matches = list(pattern.finditer(s))
    for m in reversed(matches):
        paren_pos = m.end() - 1
        parsed = _find_balanced_args(s, paren_pos)
        if parsed is None:
            continue
        args, end_pos = parsed
        replacement = handler(args)
        if replacement is not None:
            s = s[:m.start()] + replacement + s[end_pos + 1:]
    return s


def sap_to_sqlite_sql(sql: str) -> str:
    """
    Best-effort conversion of SAP HANA / ANSI SQL to SQLite-compatible SQL.
    Handles common SAP SQL idioms used in our generated queries.
    """
    s = sql.strip().rstrip(";")

    # Remove SQL single-line comments
    s = _re.sub(r'--[^\n]*', '', s)

    # If there are multiple statements, keep the last SELECT/WITH
    if ';' in s:
        stmts = []
        current = []
        in_quote = False
        for ch in s:
            if ch == "'" and not in_quote:
                in_quote = True
                current.append(ch)
            elif ch == "'" and in_quote:
                in_quote = False
                current.append(ch)
            elif ch == ';' and not in_quote:
                stmt = ''.join(current).strip()
                if stmt:
                    stmts.append(stmt)
                current = []
            else:
                current.append(ch)
        last_part = ''.join(current).strip()
        if last_part:
            stmts.append(last_part)
        if len(stmts) > 1:
            main_stmt = None
            for stmt in reversed(stmts):
                if _re.match(r'\s*(SELECT|WITH)\b', stmt, _re.IGNORECASE):
                    main_stmt = stmt
                    break
            s = main_stmt if main_stmt else stmts[-1]

    # DATE literal: DATE '2025-01-01' -> '2025-01-01'
    s = _re.sub(r"\bDATE\s+'([^']+)'", r"'\1'", s, flags=_re.IGNORECASE)

    # TIMESTAMP literal: TIMESTAMP '2025-01-01 00:00:00' -> '2025-01-01 00:00:00'
    s = _re.sub(r"\bTIMESTAMP\s+'([^']+)'", r"'\1'", s, flags=_re.IGNORECASE)

    # CURRENT_DATE -> date('now')
    s = _re.sub(r'\bCURRENT_DATE\b', "date('now')", s, flags=_re.IGNORECASE)

    # CURRENT_TIMESTAMP -> datetime('now')
    s = _re.sub(r'\bCURRENT_TIMESTAMP\b', "datetime('now')", s, flags=_re.IGNORECASE)

    # IFNULL -> COALESCE (SQLite supports both, but be consistent)
    s = _re.sub(r'\bIFNULL\s*\(', 'COALESCE(', s, flags=_re.IGNORECASE)

    # NVL -> COALESCE (in case LLM uses Oracle-style)
    s = _re.sub(r'\bNVL\s*\(', 'COALESCE(', s, flags=_re.IGNORECASE)

    # SYSDATE -> date('now') (in case LLM uses Oracle-style)
    s = _re.sub(r'\bSYSDATE\b', "date('now')", s, flags=_re.IGNORECASE)

    # TO_VARCHAR / TO_CHAR -> handle SAP HANA style
    def _to_varchar_handler(args):
        if len(args) == 1:
            return f"CAST({args[0].strip()} AS TEXT)"
        if len(args) >= 2:
            expr = args[0].strip()
            fmt = args[1].strip().strip("'\"")
            fmt_upper = fmt.upper()
            # Number format
            is_number_fmt = bool(_re.search(r'[90]', fmt)) and not _re.search(r'(YYYY|YY|MM|DD)', fmt_upper)
            if is_number_fmt:
                decimal_match = _re.search(r'\.([09]+)', fmt)
                if decimal_match:
                    decimals = len(decimal_match.group(1))
                    return f"printf('%.{decimals}f', {expr})"
                return f"CAST({expr} AS TEXT)"
            # Date format
            sqlite_fmt = fmt
            fmt_map = {
                'YYYY': '%Y', 'YY': '%y', 'MM': '%m', 'DD': '%d',
                'MON': '%m', 'MONTH': '%m', 'HH24': '%H', 'MI': '%M', 'SS': '%S',
            }
            for ofmt, sfmt in fmt_map.items():
                sqlite_fmt = sqlite_fmt.replace(ofmt, sfmt)
            return f"strftime('{sqlite_fmt}', {expr})"
        return None
    s = _replace_function_calls(s, 'TO_VARCHAR', _to_varchar_handler)
    s = _replace_function_calls(s, 'TO_CHAR', _to_varchar_handler)

    # TO_DATE('...', 'fmt') -> just the date string
    def _to_date_handler(args):
        if len(args) >= 1:
            val = args[0].strip().strip("'\"")
            return f"'{val}'"
        return None
    s = _replace_function_calls(s, 'TO_DATE', _to_date_handler)

    # TO_NUMBER(...) -> CAST(... AS REAL)
    def _to_number_handler(args):
        if len(args) >= 1:
            return f"CAST({args[0].strip()} AS REAL)"
        return None
    s = _replace_function_calls(s, 'TO_NUMBER', _to_number_handler)
    s = _replace_function_calls(s, 'TO_INTEGER', _to_number_handler)
    s = _replace_function_calls(s, 'TO_INT', _to_number_handler)

    # CONCAT(a, b) -> (a || b) — SAP HANA uses CONCAT, SQLite uses ||
    def _concat_handler(args):
        if len(args) >= 2:
            return '(' + ' || '.join(a.strip() for a in args) + ')'
        elif len(args) == 1:
            return args[0].strip()
        return None
    s = _replace_function_calls(s, 'CONCAT', _concat_handler)

    # LPAD(expr, n, 'c') -> supported in newer SQLite but let's use printf for safety
    def _lpad_handler(args):
        if len(args) >= 2:
            expr = args[0].strip()
            width = args[1].strip()
            pad_char = args[2].strip().strip("'\"") if len(args) >= 3 else '0'
            if pad_char == '0':
                return f"printf('%0{width}s', {expr})"
            return f"printf('%{width}s', {expr})"
        return None
    s = _replace_function_calls(s, 'LPAD', _lpad_handler)

    # SUBSTRING -> SUBSTR (SQLite uses SUBSTR)
    s = _re.sub(r'\bSUBSTRING\s*\(', 'SUBSTR(', s, flags=_re.IGNORECASE)

    # EXTRACT(YEAR|MONTH|DAY FROM expr) -> CAST(strftime(...) AS INTEGER)
    def _extract_handler(args):
        if len(args) != 1:
            return None
        inner = args[0].strip()
        m = _re.match(r'(YEAR|MONTH|DAY|HOUR|MINUTE|SECOND)\s+FROM\s+(.+)',
                       inner, _re.IGNORECASE | _re.DOTALL)
        if not m:
            return None
        part = m.group(1).upper()
        expr = m.group(2).strip()
        fmt_map = {'YEAR': '%Y', 'MONTH': '%m', 'DAY': '%d',
                   'HOUR': '%H', 'MINUTE': '%M', 'SECOND': '%S'}
        fmt = fmt_map.get(part, '%Y')
        return f"CAST(strftime('{fmt}', {expr}) AS INTEGER)"
    s = _replace_function_calls(s, 'EXTRACT', _extract_handler)

    # ADD_DAYS(date, n) -> date(date, '+n days')
    def _add_days_handler(args):
        if len(args) == 2:
            date_expr = args[0].strip()
            days = args[1].strip()
            if days.startswith('-'):
                return f"date({date_expr}, '{days} days')"
            else:
                return f"date({date_expr}, '+{days} days')"
        return None
    s = _replace_function_calls(s, 'ADD_DAYS', _add_days_handler)

    # ADD_MONTHS(date, n) -> date(date, '+n months')
    def _add_months_handler(args):
        if len(args) == 2:
            date_expr = args[0].strip()
            months = args[1].strip()
            if months.startswith('-'):
                return f"date({date_expr}, '{months} months')"
            else:
                return f"date({date_expr}, '+{months} months')"
        return None
    s = _replace_function_calls(s, 'ADD_MONTHS', _add_months_handler)

    # DATEDIFF(day, a, b) -> CAST((julianday(b) - julianday(a)) AS INTEGER)
    # SQL Server / HANA style: DATEDIFF(interval, start, end)
    def _datediff_handler(args):
        if len(args) == 3:
            # args[0] is the interval (day, month, etc.), args[1] start, args[2] end
            return f"CAST((julianday({args[2].strip()}) - julianday({args[1].strip()})) AS INTEGER)"
        if len(args) == 2:
            # Fallback: DATEDIFF(a, b) without interval
            return f"CAST((julianday({args[0].strip()}) - julianday({args[1].strip()})) AS INTEGER)"
        return None
    s = _replace_function_calls(s, 'DATEDIFF', _datediff_handler)

    # DAYS_BETWEEN(a, b) -> (julianday(a) - julianday(b))
    def _days_between_handler(args):
        if len(args) == 2:
            return f"(julianday({args[0].strip()}) - julianday({args[1].strip()}))"
        return None
    s = _replace_function_calls(s, 'DAYS_BETWEEN', _days_between_handler)

    # MONTHS_BETWEEN(a,b) -> (julianday(a) - julianday(b))/30.44
    def _months_between_handler(args):
        if len(args) == 2:
            return f"((julianday({args[0].strip()}) - julianday({args[1].strip()})) / 30.44)"
        return None
    s = _replace_function_calls(s, 'MONTHS_BETWEEN', _months_between_handler)

    # STDDEV / VARIANCE — not available in SQLite; strip them
    s = _re.sub(r',?\s*STDDEV\s*\([^)]*\)\s*(?:AS\s+\w+)?', '', s, flags=_re.IGNORECASE)
    s = _re.sub(r',?\s*VARIANCE\s*\([^)]*\)\s*(?:AS\s+\w+)?', '', s, flags=_re.IGNORECASE)
    s = _re.sub(r'SELECT\s*,', 'SELECT ', s, flags=_re.IGNORECASE)

    # TOP n -> (will be handled by adding LIMIT at end)
    top_match = _re.match(r'(SELECT)\s+TOP\s+(\d+)\s+', s, _re.IGNORECASE)
    if top_match:
        limit_val = top_match.group(2)
        s = top_match.group(1) + ' ' + s[top_match.end():]
        if not _re.search(r'\bLIMIT\b', s, _re.IGNORECASE):
            s += f" LIMIT {limit_val}"

    # FETCH FIRST n ROWS ONLY -> LIMIT n
    fetch_match = _re.search(r'FETCH\s+(?:FIRST|NEXT)\s+(\d+)\s+ROWS?\s+ONLY', s, _re.IGNORECASE)
    if fetch_match:
        s = s[:fetch_match.start()] + f"LIMIT {fetch_match.group(1)}"

    # ROWNUM handling (Oracle-style, in case LLM generates it)
    s = _re.sub(r"\bAND\s+ROWNUM\s*<=\s*(\d+)", '', s, flags=_re.IGNORECASE)
    s = _re.sub(r"\bWHERE\s+ROWNUM\s*<=\s*(\d+)", '', s, flags=_re.IGNORECASE)

    # TRUNC (in case LLM uses it)
    def _trunc_handler(args):
        if len(args) == 2:
            expr = args[0].strip()
            fmt = args[1].strip().strip("'\"").upper()
            if fmt in ('MM', 'MON', 'MONTH'):
                return f"date({expr}, 'start of month')"
            elif fmt in ('YYYY', 'YY', 'YEAR'):
                return f"date({expr}, 'start of year')"
            else:
                return f"date({expr})"
        elif len(args) == 1:
            return f"date({args[0].strip()})"
        return None
    s = _replace_function_calls(s, 'TRUNC', _trunc_handler)

    # String concat || is natively supported in SQLite — no change needed

    return s


def _check_pii_violation(sql: str) -> str | None:
    """
    Hard-coded PII enforcement.  If the SQL touches any HR / PAY / BEN
    table WITHOUT proper aggregation, return an error message.
    Returns None when the query is safe to execute.
    """
    upper = sql.upper()

    # SAP HR/PAY/BEN tables that contain PII
    PII_TABLES = [
        "PA0001", "PA0002", "PA0006", "PA0008", "PA0014",
        "PA0167", "PA0168", "PA0169",
        "HRPY_RGDIR", "T511", "T512T",
    ]

    # Does the query reference any PII table?
    touched = [t for t in PII_TABLES if t in upper]
    if not touched:
        return None                       # no PII tables → safe

    # Require GROUP BY
    if "GROUP BY" not in upper:
        return (
            "PII Protection: queries against HR / Payroll / Benefits tables "
            f"({', '.join(touched)}) must aggregate results with GROUP BY. "
            "Individual employee data cannot be returned. "
            "Try an aggregate query such as headcount by org unit or "
            "average salary by department."
        )

    # Require at least one aggregate function in the SELECT clause
    select_part = upper.split("FROM")[0] if "FROM" in upper else upper
    AGG_FUNCS = ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]
    if not any(fn in select_part for fn in AGG_FUNCS):
        return (
            "PII Protection: queries against HR / Payroll / Benefits tables "
            f"({', '.join(touched)}) must use aggregate functions "
            "(COUNT, SUM, AVG, MIN, MAX) in the SELECT clause. "
            "Individual employee data cannot be returned."
        )

    # Block PII columns in SELECT — but allow them inside aggregate functions
    PII_COLUMNS = ["PERNR", "ENAME", "NACHN", "VORNA", "GBDAT", "PERID"]
    # Strip out aggregate-wrapped expressions before checking
    import re as _pii_re
    stripped_select = _pii_re.sub(
        r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\([^)]*\)', '', select_part
    )
    selected_pii = [c for c in PII_COLUMNS if c in stripped_select]
    if selected_pii:
        return (
            "PII Protection: the SELECT clause includes personally identifiable "
            f"columns ({', '.join(selected_pii)}). Queries against HR / Payroll / "
            "Benefits tables must not return individual employee identifiers. "
            "Use GROUP BY on organizational attributes and aggregate functions instead."
        )

    return None                           # passed all checks


def execute_on_test_db(sql: str, max_rows: int = 200) -> dict:
    """
    Execute a SQL query against the test SQLite database.
    Returns {"columns": [...], "rows": [...], "row_count": n} or {"error": "..."}.
    """
    if not os.path.exists(TEST_DB_FILE):
        return {"error": "Test database not found. Place sap_test.db in the application folder."}

    # ── PII gate — hard block before any execution ──
    pii_err = _check_pii_violation(sql)
    if pii_err:
        return {"error": pii_err}

    # Convert SAP SQL to SQLite
    sqlite_sql = sap_to_sqlite_sql(sql)

    try:
        conn = sqlite3.connect(TEST_DB_FILE)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        cursor.execute(sqlite_sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows)
        total = len(rows)
        extra = cursor.fetchone()
        has_more = extra is not None
        conn.close()

        # Detect column types from actual data
        num_cols = len(columns)
        col_types = ["str"] * num_cols
        for row in rows:
            for ci in range(num_cols):
                v = row[ci]
                if v is None:
                    continue
                if isinstance(v, float):
                    col_types[ci] = "float"
                elif isinstance(v, int) and not isinstance(v, bool):
                    if col_types[ci] != "float":
                        col_types[ci] = "int"

        # Promote int to float when mixed with floats
        has_float = any(t == "float" for t in col_types)
        has_int = any(t == "int" for t in col_types)
        if has_float and has_int:
            for ci in range(num_cols):
                if col_types[ci] == "int":
                    col_types[ci] = "float"

        # Convert to serializable types
        clean_rows = []
        for row in rows:
            clean_row = []
            for v in row:
                if v is None:
                    clean_row.append(None)
                elif isinstance(v, bool):
                    clean_row.append(v)
                elif isinstance(v, int):
                    clean_row.append(v)
                elif isinstance(v, float):
                    clean_row.append(v)
                else:
                    clean_row.append(str(v))
            clean_rows.append(clean_row)

        return {
            "columns": columns,
            "column_types": col_types,
            "rows": clean_rows,
            "row_count": total,
            "has_more": has_more,
            "sqlite_sql": sqlite_sql
        }
    except Exception as e:
        return {"error": str(e), "sqlite_sql": sqlite_sql}


# ---------------------------------------------------------------------------
# HTTP Server (powers the web UI)
# ---------------------------------------------------------------------------
class SAPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the SAP SQL Engine web UI."""

    system_prompt = ""
    config = {}

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_html(self, path):
        if os.path.exists(path):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(HTML_FILE)

        elif self.path == "/api/status":
            api_key = get_api_key(self.config)
            self._send_json({
                "has_api": bool(api_key),
                "model": self.config.get("model", CLAUDE_MODEL),
                "key_preview": f"...{api_key[-6:]}" if len(api_key) > 6 else "",
                "has_test_db": os.path.exists(TEST_DB_FILE)
            })
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length).decode()) if content_length else {}

        if self.path == "/api/generate":
            question = body.get("question", "")
            conversation_history = body.get("history", [])
            api_key = get_api_key(self.config)

            if not api_key:
                self._send_json({"status": "error",
                                 "error": "No API key configured. Click the settings icon to add one."})
                return

            try:
                api_result = generate_sql_with_api(
                    question, self.system_prompt, api_key,
                    self.config.get("model", CLAUDE_MODEL),
                    conversation_history=conversation_history
                )
                self._send_json({"status": "ok", "result": api_result["text"],
                                 "usage": api_result["usage"], "mode": "api"})
            except Exception as e:
                err_msg = str(e) or f"{type(e).__name__}: {repr(e)}"
                self._send_json({"status": "error", "error": err_msg})

        elif self.path == "/api/save-key":
            key = body.get("key", "").strip()
            if not key.startswith("sk-ant-"):
                self._send_json({"status": "error",
                                 "error": "Invalid key format. Should start with sk-ant-"})
                return

            try:
                import anthropic
                try:
                    import httpx
                    http_client = httpx.Client(verify=False)
                    client = anthropic.Anthropic(api_key=key, http_client=http_client)
                except ImportError:
                    client = anthropic.Anthropic(api_key=key)
                client.messages.create(
                    model=self.config.get("model", CLAUDE_MODEL),
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Say OK"}]
                )
            except Exception as e:
                err = str(e)
                if "authentication" in err.lower() or "invalid" in err.lower() or "api key" in err.lower():
                    self._send_json({"status": "error",
                                     "error": "API key validation failed. Check that the key is correct and has credits."})
                    return
                if "credit" in err.lower() or "balance" in err.lower():
                    self._send_json({"status": "error",
                                     "error": "API key has insufficient credits. Please add credits at console.anthropic.com."})
                    return
                pass

            self.config["anthropic_api_key"] = key
            save_config(self.config)
            self._send_json({"status": "ok", "key_preview": f"...{key[-6:]}"})

        elif self.path == "/api/remove-key":
            self.config["anthropic_api_key"] = ""
            save_config(self.config)
            self._send_json({"status": "ok"})

        elif self.path == "/api/save-model":
            model = body.get("model", "").strip()
            ALLOWED_MODELS = {
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4-20250514",
                "claude-haiku-4-20250414",
            }
            if model not in ALLOWED_MODELS:
                self._send_json({"status": "error",
                                 "error": f"Unknown model: {model}"})
                return
            self.config["model"] = model
            save_config(self.config)
            self._send_json({"status": "ok", "model": model})

        elif self.path == "/api/execute":
            sql = body.get("sql", "").strip()
            if not sql:
                self._send_json({"status": "error", "error": "No SQL provided."})
                return
            result = execute_on_test_db(sql)
            if "error" in result:
                self._send_json({"status": "error", **result})
            else:
                self._send_json({"status": "ok", **result})

        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        if args and "404" in str(args[0]):
            super().log_message(format, *args)


def run_server(system_prompt: str, config: dict, open_browser: bool = True):
    """Start the HTTP server and optionally open the browser."""
    port = int(os.environ.get("PORT", config.get("port", DEFAULT_PORT)))
    host = os.environ.get("HOST", "127.0.0.1")

    SAPHandler.system_prompt = system_prompt
    SAPHandler.config = config

    server = HTTPServer((host, port), SAPHandler)
    url = f"http://{host}:{port}"

    api_key = get_api_key(config)
    print()
    print("=" * 56)
    print("  SAP ECC 6.0 Prompt-to-SQL Engine")
    print("=" * 56)
    print(f"  URL:    {url}")
    print(f"  Model:  {config.get('model', CLAUDE_MODEL)}")
    if api_key:
        print(f"  API:    Connected (...{api_key[-6:]})")
    else:
        print(f"  API:    Not configured (set in browser)")
    print(f"\n  Press Ctrl+C to stop")
    print("=" * 56)

    if open_browser and not os.environ.get("PORT"):
        def _open():
            time.sleep(0.8)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------
def run_interactive(system_prompt: str, config: dict):
    api_key = get_api_key(config)
    has_api = bool(api_key)

    print("\n" + "=" * 60)
    print("  SAP ECC 6.0 Prompt-to-SQL Engine (CLI)")
    print("  Modules: FI-GL, FI-AP, FI-AR, CO, MM, SD, HR, PAY, BEN")
    print("=" * 60)
    if has_api:
        print(f"  Mode: API (Claude {config.get('model', CLAUDE_MODEL)})")
    else:
        print("  Mode: No API key — use 'key <your-key>' to set one")
    print("\n  Commands: 'examples', 'key <api-key>', 'export', 'web', 'quit'")
    print("=" * 60)

    examples = [
        "Who are our top 10 vendors by total spend in company code 1000?",
        "Show me all open vendor items (unpaid invoices) for vendor Acme Corp",
        "What is the GL trial balance for company 1000 in fiscal year 2025?",
        "Show me all purchase orders over $50,000 that are still open",
        "What sales orders were created in Q1 2025 and what is their delivery status?",
        "Show me cost center actual vs plan for cost center 1000 in 2025",
        "List all employees in personnel area 1000 with their positions",
        "What journal entries were posted to account 400000 in January 2025?",
        "Show me customer open items (outstanding receivables) by aging bucket",
        "What materials have inventory below reorder point?",
    ]

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "examples":
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. {ex}")
            continue
        if question.lower().startswith("key "):
            new_key = question[4:].strip()
            config["anthropic_api_key"] = new_key
            save_config(config)
            api_key = new_key
            has_api = True
            print(f"  API key saved (...{new_key[-6:]})")
            continue
        if question.lower() == "export":
            out = os.path.join(APP_DIR, "sap_system_prompt.txt")
            with open(out, "w") as f:
                f.write(system_prompt)
            print(f"  Exported to: {out}")
            continue
        if question.lower() == "web":
            run_server(system_prompt, config)
            break

        if question.isdigit() and 1 <= int(question) <= len(examples):
            question = examples[int(question) - 1]
            print(f"  -> {question}")

        api_key = get_api_key(config)
        if api_key:
            print("\n  Generating SQL...\n")
            try:
                api_result = generate_sql_with_api(question, system_prompt, api_key,
                                               config.get("model", CLAUDE_MODEL))
                print(api_result["text"])
                print(f"\n  Tokens: {api_result['usage']['input_tokens']} input, {api_result['usage']['output_tokens']} output")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print("  No API key. Use 'key <your-key>' or 'web' to open the browser UI.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SAP ECC 6.0 Prompt-to-SQL Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Quick start:
          python3 sap_sql_engine.py --server        # Web UI (recommended)
          python3 sap_sql_engine.py                  # Interactive CLI
          python3 sap_sql_engine.py -q "question"    # Single question
        """)
    )
    parser.add_argument("--server", action="store_true", help="Start web UI (recommended)")
    parser.add_argument("-q", "--question", help="Single question mode")
    parser.add_argument("--port", type=int, help="Port for web UI (default: 8766)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--export-system-prompt", action="store_true",
                        help="Export system prompt to file")
    args = parser.parse_args()

    ensure_anthropic_installed()

    config = load_config()
    if args.port:
        config["port"] = args.port

    try:
        model = load_semantic_model()
    except FileNotFoundError:
        print(f"Error: Semantic model not found at {MODEL_FILE}")
        print("Make sure sap_semantic_model.json is in the same folder.")
        sys.exit(1)

    system_prompt = build_system_prompt(model)

    if args.export_system_prompt:
        out = os.path.join(APP_DIR, "sap_system_prompt.txt")
        with open(out, "w") as f:
            f.write(system_prompt)
        print(f"Exported to: {out} ({len(system_prompt):,} chars)")

    elif args.question:
        api_key = get_api_key(config)
        if not api_key:
            print("No API key found. Set one with:")
            print("  export ANTHROPIC_API_KEY=sk-ant-...")
            print("  — or —")
            print("  python3 sap_sql_engine.py --server  (then configure in browser)")
            sys.exit(1)
        try:
            api_result = generate_sql_with_api(args.question, system_prompt, api_key,
                                        config.get("model", CLAUDE_MODEL))
            print(api_result["text"])
            print(f"\nTokens: {api_result['usage']['input_tokens']} input, {api_result['usage']['output_tokens']} output")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.server:
        run_server(system_prompt, config, open_browser=not args.no_browser)

    else:
        run_interactive(system_prompt, config)


if __name__ == "__main__":
    main()
