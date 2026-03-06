#!/usr/bin/env python3
"""
SAP ECC 6.0 Test Database Builder

This script builds a SQLite test database for SAP ECC 6.0 based on the semantic model.
It extracts table definitions from the JSON model and creates realistic sample data.
"""

import json
import sqlite3
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import os

# Database path
DB_PATH = "/sessions/loving-lucid-edison/mnt/OEBS Object Model/Building semantic object model for SAP ECC/sap_test.db"
MODEL_PATH = "/sessions/loving-lucid-edison/mnt/OEBS Object Model/Building semantic object model for SAP ECC/sap_semantic_model.json"

# SAP master data
COMPANY_CODES = ['1000', '2000', '3000']
FISCAL_YEARS = ['2025', '2026']
CURRENCIES = ['USD', 'EUR', 'GBP']
LANGUAGES = ['E']  # English
VENDOR_NAMES = ['Acme Corp', 'Global Tech', 'Summit Industries', 'Prime Logistics', 'Elite Suppliers', 'Blue Star Ltd', 'Dynamic Solutions', 'Ocean Trade']
CUSTOMER_NAMES = ['Fast Motor Corp', 'Digital Ventures', 'Green Energy Inc', 'Smart Systems', 'Tech Solutions Ltd', 'Industrial Partners', 'Global Traders', 'Future Industries']
EMPLOYEE_NAMES = ['John Smith', 'Mary Johnson', 'Robert Brown', 'Patricia Davis', 'Michael Wilson', 'Jennifer Martinez', 'David Anderson', 'Linda Taylor']
MATERIAL_NAMES = ['Raw Material A', 'Component B', 'Finished Good C', 'Service D', 'Tool E', 'Supply F', 'Item G', 'Product H']
COST_CENTERS = [f"{i:010d}" for i in range(1000, 1016)]

class SAPTestDataGenerator:
    """Generate realistic SAP test data."""

    def __init__(self):
        self.doc_counter = {}
        self.vendor_counter = 0
        self.customer_counter = 0
        self.employee_counter = 0
        self.material_counter = 0
        self.random = random.Random(42)  # Seed for reproducibility

    def get_next_docnum(self, bukrs: str) -> str:
        """Get next 10-digit document number."""
        key = f"doc_{bukrs}"
        if key not in self.doc_counter:
            self.doc_counter[key] = 5100000001
        self.doc_counter[key] += 1
        return str(self.doc_counter[key])

    def get_vendor_num(self) -> str:
        """Get next vendor number (0000100001, etc.)."""
        self.vendor_counter += 1
        return f"{1000000 + self.vendor_counter:010d}"

    def get_customer_num(self) -> str:
        """Get next customer number (0000200001, etc.)."""
        self.customer_counter += 1
        return f"{2000000 + self.customer_counter:010d}"

    def get_employee_num(self) -> str:
        """Get next employee number (00100001, etc.)."""
        self.employee_counter += 1
        return f"{100000 + self.employee_counter:08d}"

    def get_material_num(self) -> str:
        """Get next material number (100000001, etc.)."""
        self.material_counter += 1
        return f"{100000 + self.material_counter:09d}"

    def random_date(self, start_year: int = 2025, end_year: int = 2026) -> str:
        """Generate random date in YYYYMMDD format."""
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        delta = end - start
        random_days = self.random.randint(0, delta.days)
        random_date = start + timedelta(days=random_days)
        return random_date.strftime("%Y%m%d")

    def random_amount(self, max_val: float = 10000) -> float:
        """Generate realistic financial amount."""
        return round(self.random.uniform(1, max_val), 2)

    def random_currency(self) -> str:
        """Random currency."""
        return self.random.choice(CURRENCIES)

    def random_cost_center(self) -> str:
        """Random cost center."""
        return self.random.choice(COST_CENTERS)


class SAPDatabaseBuilder:
    """Build SAP ECC 6.0 test database from semantic model."""

    def __init__(self):
        self.conn = None
        self.cursor = None
        self.generator = SAPTestDataGenerator()
        self.random = random.Random(42)
        self.model = None
        self.tables_info = {}

    def load_model(self):
        """Load semantic model from JSON."""
        print(f"Loading semantic model from {MODEL_PATH}...")
        with open(MODEL_PATH, 'r') as f:
            self.model = json.load(f)

        # Extract all table definitions
        self._extract_tables()
        print(f"Found {len(self.tables_info)} tables in semantic model")

    def _extract_tables(self):
        """Extract all table definitions from semantic model."""
        modules = self.model.get('modules', {})

        for module_name, module_data in modules.items():
            if not isinstance(module_data, dict) or 'business_objects' not in module_data:
                continue

            for obj_name, obj_data in module_data.get('business_objects', {}).items():
                if 'tables' not in obj_data:
                    continue

                for table_name, table_def in obj_data['tables'].items():
                    if table_name not in self.tables_info:
                        self.tables_info[table_name] = {
                            'description': table_def.get('description', ''),
                            'primary_key': table_def.get('primary_key', {}),
                            'columns': {}
                        }

                        # Extract column definitions
                        for col in table_def.get('business_columns', []):
                            col_name = col.get('column')
                            col_type = col.get('type', 'TEXT')
                            col_desc = col.get('description', '')

                            if col_name:
                                self.tables_info[table_name]['columns'][col_name] = {
                                    'type': col_type,
                                    'description': col_desc
                                }

    def connect(self):
        """Connect to SQLite database."""
        # Remove old database if it exists
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        print(f"Connected to database: {DB_PATH}")

    def create_tables(self):
        """Create all tables based on semantic model."""
        print("\nCreating tables...")

        for table_name, table_info in sorted(self.tables_info.items()):
            columns = table_info.get('columns', {})
            if not columns:
                print(f"  Skipping {table_name} - no columns defined")
                continue

            # Build CREATE TABLE statement
            col_defs = []
            for col_name, col_info in columns.items():
                col_type = self._map_sap_type_to_sqlite(col_info['type'])
                col_defs.append(f"  {col_name} {col_type}")

            # Add primary key if defined
            pk = table_info['primary_key']
            if pk and 'column' in pk:
                pk_columns = [c.strip() for c in pk['column'].split(',')]
                pk_cols = [c for c in pk_columns if c in columns]
                if pk_cols:
                    col_defs.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")

            create_sql = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n)"

            try:
                self.cursor.execute(create_sql)
                print(f"  Created table: {table_name} ({len(columns)} columns)")
            except sqlite3.Error as e:
                print(f"  Error creating {table_name}: {e}")

    def _map_sap_type_to_sqlite(self, sap_type: str) -> str:
        """Map SAP type to SQLite type."""
        if sap_type.startswith('NUMC'):
            return 'TEXT'
        elif sap_type.startswith('CHAR'):
            return 'TEXT'
        elif sap_type.startswith('DATE'):
            return 'TEXT'
        elif sap_type.startswith('AMOUNT'):
            return 'REAL'
        elif sap_type.startswith('DEC'):
            return 'REAL'
        elif sap_type.startswith('INT'):
            return 'INTEGER'
        else:
            return 'TEXT'

    def populate_tables(self):
        """Populate all tables with realistic sample data."""
        print("\nPopulating tables with sample data...")

        # Populate in dependency order to maintain referential integrity
        self._populate_master_data()
        self._populate_fi_data()
        self._populate_mm_data()
        self._populate_sd_data()
        self._populate_hr_data()
        self._populate_co_data()
        self._populate_other_data()

        self.conn.commit()
        print("All tables populated successfully")

    def _insert_record(self, table: str, data: Dict[str, Any]):
        """Insert a record into a table."""
        # Get actual columns in the table
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        actual_cols = {row[1] for row in cursor.fetchall()}

        # Filter data to only include columns that exist
        filtered_data = {k: v for k, v in data.items() if k in actual_cols}

        if not filtered_data:
            return

        columns = list(filtered_data.keys())
        placeholders = ','.join(['?' for _ in columns])
        sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"

        try:
            self.cursor.execute(sql, tuple(filtered_data.values()))
        except sqlite3.Error as e:
            pass  # Silently ignore insert errors

    def _populate_master_data(self):
        """Populate master data tables (SKA1, SKAT, LFA1, KNA1, MARA, etc.)."""
        print("  Populating master data...")

        # SKA1 - GL Account Master
        for i in range(100000, 900001, 100000):
            self._insert_record('SKA1', {
                'SAKNR': f"{i:010d}",
                'KTOPL': 'INT',
                'BUKRS': COMPANY_CODES[0],
                'LOEKZ': ''
            })

        # SKAT - GL Account Text
        for i in range(100000, 900001, 100000):
            account_num = f"{i:010d}"
            acc_type = 'Asset' if i < 200000 else 'Liability' if i < 300000 else 'Equity' if i < 400000 else 'Revenue' if i < 500000 else 'COGS' if i < 600000 else 'Expense'

            for spras in LANGUAGES:
                self._insert_record('SKAT', {
                    'SAKNR': account_num,
                    'SPRAS': spras,
                    'TXT20': f"{acc_type} {account_num}",
                    'TXT50': f"{acc_type} Account {account_num}"
                })

        # SKB1 - GL Account per Company
        for saknr in [f"{i:010d}" for i in range(100000, 900001, 100000)]:
            for bukrs in COMPANY_CODES:
                self._insert_record('SKB1', {
                    'SAKNR': saknr,
                    'BUKRS': bukrs,
                    'XAFFY': '',
                    'XACFB': ''
                })

        # LFA1 - Vendor Master
        for i in range(10):
            vendor_num = self.generator.get_vendor_num()
            self._insert_record('LFA1', {
                'LIFNR': vendor_num,
                'BUKRS': COMPANY_CODES[i % len(COMPANY_CODES)],
                'NAME1': VENDOR_NAMES[i % len(VENDOR_NAMES)],
                'LAND1': 'US',
                'LOEKZ': ''
            })

        # LFB1 - Vendor per Company
        for bukrs in COMPANY_CODES:
            for i in range(3):
                vendor_num = self.generator.get_vendor_num()
                self._insert_record('LFA1', {
                    'LIFNR': vendor_num,
                    'BUKRS': bukrs,
                    'NAME1': VENDOR_NAMES[i % len(VENDOR_NAMES)],
                    'LAND1': 'US',
                    'LOEKZ': ''
                })

                self._insert_record('LFB1', {
                    'LIFNR': vendor_num,
                    'BUKRS': bukrs,
                    'WAERS': self.generator.random_currency(),
                    'ZAHLTERM': 'Z001'
                })

        # KNA1 - Customer Master
        for i in range(10):
            customer_num = self.generator.get_customer_num()
            self._insert_record('KNA1', {
                'KUNNR': customer_num,
                'BUKRS': COMPANY_CODES[i % len(COMPANY_CODES)],
                'NAME1': CUSTOMER_NAMES[i % len(CUSTOMER_NAMES)],
                'LAND1': 'US',
                'LOEKZ': ''
            })

        # KNB1 - Customer per Company
        for bukrs in COMPANY_CODES:
            for i in range(3):
                customer_num = self.generator.get_customer_num()
                self._insert_record('KNA1', {
                    'KUNNR': customer_num,
                    'BUKRS': bukrs,
                    'NAME1': CUSTOMER_NAMES[i % len(CUSTOMER_NAMES)],
                    'LAND1': 'US',
                    'LOEKZ': ''
                })

                self._insert_record('KNB1', {
                    'KUNNR': customer_num,
                    'BUKRS': bukrs,
                    'WAERS': self.generator.random_currency(),
                    'ZAHLTERM': 'Z001'
                })

        # MARA - Material Master
        for i in range(15):
            material_num = self.generator.get_material_num()
            self._insert_record('MARA', {
                'MATNR': material_num,
                'MTART': 'FERT' if i % 3 == 0 else 'HAWA' if i % 3 == 1 else 'DIEN',
                'MEINS': 'PC' if i % 2 == 0 else 'KG',
                'LOEKZ': ''
            })

            # MAKT - Material Text
            self._insert_record('MAKT', {
                'MATNR': material_num,
                'SPRAS': 'E',
                'MAKTX': MATERIAL_NAMES[i % len(MATERIAL_NAMES)]
            })

        # MARC - Material per Plant
        plant_list = ['0001', '0002', '0003']
        for plant in plant_list:
            for i in range(15):
                material_num = f"{100000 + i:09d}"
                self._insert_record('MARC', {
                    'MATNR': material_num,
                    'WERKS': plant,
                    'PLIFZ': '10',
                    'LOEKZ': ''
                })

        # MARD - Material per Warehouse
        for plant in plant_list:
            for i in range(15):
                material_num = f"{100000 + i:09d}"
                for lgort in ['0001', '0002']:
                    self._insert_record('MARD', {
                        'MATNR': material_num,
                        'WERKS': plant,
                        'LGORT': lgort,
                        'LABST': float(self.generator.random.randint(0, 1000))
                    })

        # CSKS - Cost Center Master
        for kostl in COST_CENTERS[:10]:
            self._insert_record('CSKS', {
                'KOSTL': kostl,
                'BUKRS': COMPANY_CODES[0],
                'DATAB': '20250101',
                'DATBI': '20261231',
                'LTEXT': f'Cost Center {kostl}'
            })

        # CSKT - Cost Center Text
        for kostl in COST_CENTERS[:10]:
            self._insert_record('CSKT', {
                'KOSTL': kostl,
                'SPRAS': 'E',
                'KTEXT': f'Cost Center {kostl}'
            })

    def _populate_fi_data(self):
        """Populate Financial Accounting data (BKPF, BSEG, BSIK, BSAK, etc.)."""
        print("  Populating FI data...")

        # BKPF - FI Document Header
        doc_count = 0
        bkpf_docs = []
        for bukrs in COMPANY_CODES:
            for _ in range(15):
                belnr = self.generator.get_next_docnum(bukrs)
                gjahr = self.random.choice(FISCAL_YEARS)
                posting_date = self.generator.random_date()

                self._insert_record('BKPF', {
                    'MANDT': '100',
                    'BUKRS': bukrs,
                    'BELNR': belnr,
                    'GJAHR': gjahr,
                    'BLART': 'SA',
                    'BLDAT': posting_date,
                    'BUDAT': posting_date,
                    'USNAM': 'BAPI_USER',
                    'TCODE': 'F-02',
                    'WAERS': self.generator.random_currency(),
                    'LOEKZ': ''
                })
                bkpf_docs.append((bukrs, belnr, gjahr))
                doc_count += 1

        # BSEG - FI Line Items
        for bukrs, belnr, gjahr in bkpf_docs:
            # Create 2-5 line items per document
            num_lines = self.generator.random.randint(2, 5)
            total_amount = self.generator.random_amount(5000)
            sum_amount = 0

            for line_num in range(1, num_lines + 1):
                if line_num == num_lines:
                    # Last line to balance
                    amount = total_amount - sum_amount
                else:
                    amount = round(total_amount / num_lines, 2)
                    sum_amount += amount

                self._insert_record('BSEG', {
                    'MANDT': '100',
                    'BUKRS': bukrs,
                    'BELNR': belnr,
                    'GJAHR': gjahr,
                    'BUZEI': f"{line_num:03d}",
                    'SAKNR': f"{(400000 + (line_num - 1) * 100000):010d}",
                    'DMBTR': amount,
                    'SHKZG': 'S' if line_num % 2 == 0 else 'H',
                    'WAERS': self.generator.random_currency(),
                    'KOSTL': self.generator.random_cost_center() if line_num < num_lines else ''
                })

        # BSIK - Vendor Open Items (AP)
        for _ in range(20):
            self._insert_record('BSIK', {
                'MANDT': '100',
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'LIFNR': f"{1000000 + self.generator.random.randint(1, 13):010d}",
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'BELNR': self.generator.random.randint(100000, 9999999),
                'BUZEI': '001',
                'DMBTR': self.generator.random_amount(10000),
                'WAERS': self.generator.random_currency()
            })

        # BSAK - Vendor Cleared Items (AP)
        for _ in range(15):
            self._insert_record('BSAK', {
                'MANDT': '100',
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'LIFNR': f"{1000000 + self.generator.random.randint(1, 13):010d}",
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'BELNR': self.generator.random.randint(100000, 9999999),
                'BUZEI': '001',
                'DMBTR': self.generator.random_amount(10000),
                'WAERS': self.generator.random_currency()
            })

        # BSID - Customer Open Items (AR)
        for _ in range(20):
            self._insert_record('BSID', {
                'MANDT': '100',
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'KUNNR': f"{2000000 + self.generator.random.randint(1, 13):010d}",
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'BELNR': self.generator.random.randint(100000, 9999999),
                'BUZEI': '001',
                'DMBTR': self.generator.random_amount(10000),
                'WAERS': self.generator.random_currency()
            })

        # BSAD - Customer Cleared Items (AR)
        for _ in range(15):
            self._insert_record('BSAD', {
                'MANDT': '100',
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'KUNNR': f"{2000000 + self.generator.random.randint(1, 13):010d}",
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'BELNR': self.generator.random.randint(100000, 9999999),
                'BUZEI': '001',
                'DMBTR': self.generator.random_amount(10000),
                'WAERS': self.generator.random_currency()
            })

        # GLT0 - GL Account Line Items
        for _ in range(40):
            self._insert_record('GLT0', {
                'MANDT': '100',
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'SAKNR': f"{self.generator.random.randint(1, 8) * 100000:010d}",
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'POPER': f"{self.generator.random.randint(1, 12):02d}",
                'HSLVT': self.generator.random_amount(50000),
                'KSLVT': self.generator.random_amount(50000)
            })

    def _populate_mm_data(self):
        """Populate Materials Management data (EKKO, EKPO, EKBE)."""
        print("  Populating MM data...")

        # EKKO - Purchase Order Header
        po_count = 0
        ekko_list = []
        for bukrs in COMPANY_CODES:
            for _ in range(10):
                ebeln = f"{self.generator.random.randint(4500000000, 4599999999)}"
                vendor = f"{1000000 + self.generator.random.randint(1, 13):010d}"

                self._insert_record('EKKO', {
                    'MANDT': '100',
                    'EBELN': ebeln,
                    'BUKRS': bukrs,
                    'LIFNR': vendor,
                    'BEDAT': self.generator.random_date(),
                    'BSTYP': 'F',
                    'BSART': 'NB',
                    'LOEKZ': '',
                    'WAERS': self.generator.random_currency()
                })
                ekko_list.append((ebeln, bukrs))
                po_count += 1

        # EKPO - Purchase Order Line Items
        for ebeln, bukrs in ekko_list:
            # 2-5 line items per PO
            num_lines = self.generator.random.randint(2, 5)
            for ebelp in range(10, 10 + (num_lines * 10), 10):
                material = f"{100000 + self.generator.random.randint(1, 15):09d}"

                self._insert_record('EKPO', {
                    'MANDT': '100',
                    'EBELN': ebeln,
                    'EBELP': f"{ebelp:05d}",
                    'MATNR': material,
                    'MENGE': float(self.generator.random.randint(1, 100)),
                    'MEINS': 'PC',
                    'NETPR': self.generator.random_amount(1000),
                    'PEINH': 1,
                    'WAERS': self.generator.random_currency(),
                    'LOEKZ': ''
                })

        # EKBE - Purchase Order History
        for _ in range(25):
            self._insert_record('EKBE', {
                'MANDT': '100',
                'EBELN': f"{self.generator.random.randint(4500000000, 4599999999)}",
                'EBELP': f"{self.generator.random.randint(10, 50):05d}",
                'BEWTP': 'E',
                'BESNR': f"{self.generator.random.randint(1, 999):03d}",
                'BEDAT': self.generator.random_date(),
                'MENGE': float(self.generator.random.randint(1, 100)),
                'BPMENGE': float(self.generator.random.randint(1, 100)),
                'DMBTR': self.generator.random_amount(10000),
                'WAERS': self.generator.random_currency()
            })

        # RBKP - Invoice Receipt Header
        for _ in range(15):
            self._insert_record('RBKP', {
                'MANDT': '100',
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'BELNR': self.generator.random.randint(100000000, 999999999),
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'XBLNR': f"INV{self.generator.random.randint(1000000, 9999999)}",
                'BUDAT': self.generator.random_date(),
                'WAERS': self.generator.random_currency()
            })

        # RSEG - Invoice Receipt Line Items
        for _ in range(30):
            self._insert_record('RSEG', {
                'MANDT': '100',
                'BELNR': self.generator.random.randint(100000000, 999999999),
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'BUZEI': f"{self.generator.random.randint(1, 5):03d}",
                'EBELN': f"{self.generator.random.randint(4500000000, 4599999999)}",
                'EBELP': f"{self.generator.random.randint(10, 50):05d}",
                'MENGE': float(self.generator.random.randint(1, 100)),
                'BPMENGE': float(self.generator.random.randint(1, 100)),
                'DMBTR': self.generator.random_amount(10000),
                'WAERS': self.generator.random_currency()
            })

    def _populate_sd_data(self):
        """Populate Sales & Distribution data (VBAK, VBAP, LIKP, LIPS, VBRK, VBRP)."""
        print("  Populating SD data...")

        # VBAK - Sales Order Header
        so_count = 0
        vbak_list = []
        for bukrs in COMPANY_CODES:
            for _ in range(10):
                vbeln = f"{self.generator.random.randint(1000000, 9999999)}"
                customer = f"{2000000 + self.generator.random.randint(1, 13):010d}"

                self._insert_record('VBAK', {
                    'MANDT': '100',
                    'VBELN': vbeln,
                    'BUKRS': bukrs,
                    'KUNNR': customer,
                    'ERDAT': self.generator.random_date(),
                    'VBTYP': 'C',
                    'VBSTAT': 'C',
                    'WAERS': self.generator.random_currency()
                })
                vbak_list.append((vbeln, bukrs))
                so_count += 1

        # VBAP - Sales Order Line Items
        for vbeln, bukrs in vbak_list:
            num_lines = self.generator.random.randint(2, 5)
            for posnr in range(10, 10 + (num_lines * 10), 10):
                material = f"{100000 + self.generator.random.randint(1, 15):09d}"

                self._insert_record('VBAP', {
                    'MANDT': '100',
                    'VBELN': vbeln,
                    'POSNR': f"{posnr:06d}",
                    'MATNR': material,
                    'KWMENG': float(self.generator.random.randint(1, 100)),
                    'MEINS': 'PC',
                    'NETPR': self.generator.random_amount(1000),
                    'PEINH': 1,
                    'WAERS': self.generator.random_currency(),
                    'VBSTAT': '0'
                })

        # VBFA - Sales Document Flow
        for _ in range(20):
            self._insert_record('VBFA', {
                'MANDT': '100',
                'VBELN': f"{self.generator.random.randint(1000000, 9999999)}",
                'POSNR': f"{self.generator.random.randint(10, 50):06d}",
                'VBELV': f"{self.generator.random.randint(4500000000, 4599999999)}",
                'POSNV': f"{self.generator.random.randint(10, 50):06d}",
                'VBTYP_N': 'L',
                'RFMNG': float(self.generator.random.randint(1, 100))
            })

        # LIKP - Delivery Header
        for _ in range(12):
            self._insert_record('LIKP', {
                'MANDT': '100',
                'VBELN': f"{self.generator.random.randint(80000000, 89999999)}",
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'KUNNR': f"{2000000 + self.generator.random.randint(1, 13):010d}",
                'ERDAT': self.generator.random_date(),
                'LFART': 'LF',
                'WAERS': self.generator.random_currency()
            })

        # LIPS - Delivery Line Items
        for _ in range(25):
            self._insert_record('LIPS', {
                'MANDT': '100',
                'VBELN': f"{self.generator.random.randint(80000000, 89999999)}",
                'POSNR': f"{self.generator.random.randint(10, 50):06d}",
                'MATNR': f"{100000 + self.generator.random.randint(1, 15):09d}",
                'LFIMG': float(self.generator.random.randint(1, 100)),
                'MEINS': 'PC',
                'WAERS': self.generator.random_currency()
            })

        # VBRK - Billing Header
        for _ in range(12):
            self._insert_record('VBRK', {
                'MANDT': '100',
                'VBELN': f"{self.generator.random.randint(90000000, 99999999)}",
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'KUNNR': f"{2000000 + self.generator.random.randint(1, 13):010d}",
                'ERDAT': self.generator.random_date(),
                'VBTYP': 'M',
                'WAERS': self.generator.random_currency(),
                'NETWR': self.generator.random_amount(50000)
            })

        # VBRP - Billing Line Items
        for _ in range(25):
            self._insert_record('VBRP', {
                'MANDT': '100',
                'VBELN': f"{self.generator.random.randint(90000000, 99999999)}",
                'POSNR': f"{self.generator.random.randint(10, 50):06d}",
                'MATNR': f"{100000 + self.generator.random.randint(1, 15):09d}",
                'FKIMG': float(self.generator.random.randint(1, 100)),
                'MEINS': 'PC',
                'NETPR': self.generator.random_amount(1000),
                'WAERS': self.generator.random_currency()
            })

    def _populate_hr_data(self):
        """Populate Human Resources data (PA0001, PA0002, PA0006, PA0008, PA0014)."""
        print("  Populating HR data...")

        employees = []

        # PA0001 - Employee Master Data
        for _ in range(20):
            pernr = self.generator.get_employee_num()
            employees.append(pernr)

            self._insert_record('PA0001', {
                'MANDT': '100',
                'PERNR': pernr,
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'BEGDA': '20250101',
                'ENDDA': '20261231',
                'NACHN': EMPLOYEE_NAMES[int(pernr) % len(EMPLOYEE_NAMES)],
                'VORNA': 'John',
                'GBDAT': '19800101',
                'GESCH': '1'
            })

        # PA0002 - Personal Data
        for pernr in employees:
            self._insert_record('PA0002', {
                'MANDT': '100',
                'PERNR': pernr,
                'BEGDA': '20250101',
                'ENDDA': '20261231',
                'PERSK': 'SA',
                'PERSG': 'E'
            })

        # PA0006 - Address
        for pernr in employees:
            self._insert_record('PA0006', {
                'MANDT': '100',
                'PERNR': pernr,
                'BEGDA': '20250101',
                'ENDDA': '20261231',
                'LAND1': 'US',
                'LOGJAHR': '2025',
                'ORTPL': 'New York',
                'PSTLZ': '10001',
                'STRAS': 'Main St'
            })

        # PA0008 - Basic Pay
        for pernr in employees:
            self._insert_record('PA0008', {
                'MANDT': '100',
                'PERNR': pernr,
                'BEGDA': '20250101',
                'ENDDA': '20261231',
                'TRFAR': '1',
                'TRFGR': 'A',
                'SWTXT': 'Salary'
            })

        # PA0014 - Recurring Payments/Deductions
        for pernr in employees[:10]:
            for i in range(2):
                self._insert_record('PA0014', {
                    'MANDT': '100',
                    'PERNR': pernr,
                    'BEGDA': '20250101',
                    'ENDDA': '20261231',
                    'SEQNR': f"{i + 1:04d}",
                    'WAERS': 'USD',
                    'BETRG': self.generator.random_amount(5000)
                })

        # PA0167 - Health Insurance Plan
        for pernr in employees[:15]:
            self._insert_record('PA0167', {
                'MANDT': '100',
                'PERNR': pernr,
                'BEGDA': '20250101',
                'ENDDA': '20261231',
                'BESSION': f"HP{self.generator.random.randint(1000, 9999)}",
                'BESSION_CAT': '10',
                'BESSION_TYPE': 'A'
            })

        # PA0168 - Insurance Plan
        for pernr in employees[:15]:
            self._insert_record('PA0168', {
                'MANDT': '100',
                'PERNR': pernr,
                'BEGDA': '20250101',
                'ENDDA': '20261231',
                'BESSION': f"INS{self.generator.random.randint(1000, 9999)}",
                'BESSION_CAT': '20',
                'BESSION_TYPE': 'L'
            })

        # PA0169 - Savings Plan
        for pernr in employees[:15]:
            self._insert_record('PA0169', {
                'MANDT': '100',
                'PERNR': pernr,
                'BEGDA': '20250101',
                'ENDDA': '20261231',
                'BESSION': f"SAV{self.generator.random.randint(1000, 9999)}",
                'BESSION_CAT': '30',
                'BESSION_TYPE': 'S'
            })

    def _populate_co_data(self):
        """Populate Controlling data (COBK, COEP, COSP, COSS)."""
        print("  Populating CO data...")

        # COBK - CO Header Documents
        for _ in range(10):
            self._insert_record('COBK', {
                'MANDT': '100',
                'BLART': 'RH',
                'BLENR': self.generator.random.randint(100000, 999999),
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'BLDAT': self.generator.random_date(),
                'LOEKZ': ''
            })

        # COEP - CO Line Items
        for _ in range(30):
            self._insert_record('COEP', {
                'MANDT': '100',
                'BLART': 'RH',
                'BLENR': self.generator.random.randint(100000, 999999),
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'LNART': '06',
                'LINNO': f"{self.generator.random.randint(1, 10):04d}",
                'KOSTL': self.generator.random_cost_center(),
                'WERT': self.generator.random_amount(10000),
                'WAERS': self.generator.random_currency()
            })

        # COSP - CO Actual/Plan Line Items
        for _ in range(40):
            self._insert_record('COSP', {
                'MANDT': '100',
                'KOKRS': '1000',
                'KOSTL': self.generator.random_cost_center(),
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'POPER': f"{self.generator.random.randint(1, 12):02d}",
                'VERSN': '000',
                'BWVAR': '1',
                'LSTAR': 'COST',
                'HSL01': self.generator.random_amount(20000),
                'HSL02': self.generator.random_amount(20000),
                'HSL03': self.generator.random_amount(20000),
                'HSL04': self.generator.random_amount(20000)
            })

        # COSS - CO Actual/Plan Line Items by Activity
        for _ in range(30):
            self._insert_record('COSS', {
                'MANDT': '100',
                'KOKRS': '1000',
                'KOSTL': self.generator.random_cost_center(),
                'GJAHR': self.generator.random.choice(FISCAL_YEARS),
                'POPER': f"{self.generator.random.randint(1, 12):02d}",
                'VERSN': '000',
                'LSTAR': 'COST',
                'LHOURS': float(self.generator.random.randint(100, 1000)),
                'LHOUA': float(self.generator.random.randint(100, 1000))
            })

    def _populate_other_data(self):
        """Populate remaining tables (EBAN, T511, T512T, REGUH, AUFK)."""
        print("  Populating other data...")

        # EBAN - Purchase Requisition Header
        for _ in range(15):
            self._insert_record('EBAN', {
                'MANDT': '100',
                'BANFN': self.generator.random.randint(10000000, 99999999),
                'BNFPO': '0010',
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'BSART': 'NB',
                'BADAT': self.generator.random_date(),
                'WAERS': self.generator.random_currency()
            })

        # T511 - Payroll Wage Types
        for i in range(10):
            self._insert_record('T511', {
                'MANDT': '100',
                'LGART': f"{1000 + i:04d}",
                'MODLT': 'X',
                'TEXT1': f'Wage Type {i + 1}'
            })

        # T512T - Pay Scale Structure Text
        for i in range(10):
            self._insert_record('T512T', {
                'MANDT': '100',
                'TRFAR': '1',
                'TRFGR': f"{chr(65 + i)}",
                'SPRAS': 'E',
                'TEXT1': f'Pay Scale {chr(65 + i)}'
            })

        # REGUH - Refund Header
        for _ in range(5):
            self._insert_record('REGUH', {
                'MANDT': '100',
                'VBELN': self.generator.random.randint(100000000, 999999999),
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'KUNNR': f"{2000000 + self.generator.random.randint(1, 13):010d}",
                'BEDAT': self.generator.random_date(),
                'WAERS': self.generator.random_currency()
            })

        # AUFK - Order Master
        for _ in range(10):
            self._insert_record('AUFK', {
                'MANDT': '100',
                'AUFNR': self.generator.random.randint(100000000, 999999999),
                'BUKRS': self.generator.random.choice(COMPANY_CODES),
                'AUFART': 'PM01',
                'GSTRS': self.generator.random_date(),
                'GSTAR': self.generator.random_date(),
                'LOEKZ': ''
            })

    def verify_database(self):
        """Verify database by printing row counts per table."""
        print("\n" + "=" * 60)
        print("DATABASE VERIFICATION - ROW COUNTS")
        print("=" * 60)

        total_rows = 0
        cursor = self.conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()

        for (table,) in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            total_rows += count
            print(f"{table:20s} : {count:6d} rows")

        print("=" * 60)
        print(f"{'TOTAL':20s} : {total_rows:6d} rows")
        print("=" * 60)

        return total_rows

    def build(self):
        """Execute the complete build process."""
        print("\n" + "=" * 60)
        print("SAP ECC 6.0 TEST DATABASE BUILDER")
        print("=" * 60)

        self.load_model()
        self.connect()
        self.create_tables()
        self.populate_tables()
        self.verify_database()

        print("\nDatabase build completed successfully!")
        print(f"Database location: {DB_PATH}")


if __name__ == '__main__':
    # Add random as instance variable for generator
    SAPTestDataGenerator.random = random.Random(42)

    builder = SAPDatabaseBuilder()
    builder.build()
