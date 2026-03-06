#!/usr/bin/env python3
"""
SAP ECC Test Database Expander

This script expands the existing SQLite test database with additional realistic data
to support demonstrating three T-codes (FBL5N, KSB1, IW49N) and three value chains
(O2C, P2P, R2R).

It ADDS data to existing tables without recreating them.
"""

import sqlite3
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

DB_PATH = "/sessions/loving-lucid-edison/mnt/Claud/SAP ECC Semantic Model/sap_test.db"

class SAPDatabaseExpander:
    """Expand existing SAP test database with additional data."""

    def __init__(self):
        self.conn = None
        self.cursor = None
        self.random = random.Random(42)

        # Counters for generating unique numbers
        self.doc_counter = {}
        self.vendor_counter = 10000  # Start after existing vendors
        self.customer_counter = 10000  # Start after existing customers
        self.po_counter = 4600000000
        self.so_counter = 5000000
        self.delivery_counter = 80000000
        self.invoice_counter = 90000000
        self.wo_counter = 100000000
        self.gr_counter = 1000

        # Existing master data (will fetch from DB)
        self.customers = []
        self.vendors = []
        self.cost_centers = []
        self.materials = []
        self.gl_accounts = []

    def connect(self):
        """Connect to existing SQLite database."""
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        print(f"Connected to existing database: {DB_PATH}")

    def load_master_data(self):
        """Load existing master data from database."""
        print("\nLoading existing master data...")

        # Load customers
        self.cursor.execute("SELECT KUNNR FROM KNA1")
        self.customers = [row[0] for row in self.cursor.fetchall()]
        print(f"  Found {len(self.customers)} customers")

        # Load vendors
        self.cursor.execute("SELECT LIFNR FROM LFA1")
        self.vendors = [row[0] for row in self.cursor.fetchall()]
        print(f"  Found {len(self.vendors)} vendors")

        # Load cost centers
        self.cursor.execute("SELECT KOSTL FROM CSKS")
        self.cost_centers = [row[0] for row in self.cursor.fetchall()]
        print(f"  Found {len(self.cost_centers)} cost centers")

        # Load materials
        self.cursor.execute("SELECT MATNR FROM MARA")
        self.materials = [row[0] for row in self.cursor.fetchall()]
        print(f"  Found {len(self.materials)} materials")

        # Load GL accounts
        self.cursor.execute("SELECT SAKNR FROM SKA1")
        self.gl_accounts = [row[0] for row in self.cursor.fetchall()]
        print(f"  Found {len(self.gl_accounts)} GL accounts")

    def _insert_record(self, table: str, data: Dict) -> bool:
        """Insert a record into a table, silently handling errors."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            actual_cols = {row[1] for row in cursor.fetchall()}

            filtered_data = {k: v for k, v in data.items() if k in actual_cols}
            if not filtered_data:
                return False

            columns = list(filtered_data.keys())
            placeholders = ','.join(['?' for _ in columns])
            sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"

            self.cursor.execute(sql, tuple(filtered_data.values()))
            return True
        except sqlite3.Error:
            return False

    def random_date(self, days_ago_min: int = 0, days_ago_max: int = 365) -> str:
        """Generate random date in YYYYMMDD format relative to current date."""
        today = datetime(2026, 3, 6)  # Current date context
        delta_days = self.random.randint(days_ago_min, days_ago_max)
        random_date = today - timedelta(days=delta_days)
        return random_date.strftime("%Y%m%d")

    def random_amount(self, min_val: float = 500, max_val: float = 500000) -> float:
        """Generate realistic financial amount."""
        return round(self.random.uniform(min_val, max_val), 2)

    def expand_fbl5n_data(self):
        """
        Expand FBL5N - Customer Open Items report data.
        Create diverse aging buckets of customer receivables.
        """
        print("\n=== Expanding FBL5N - Customer Open Items ===")

        if not self.customers:
            print("  WARNING: No customers found, skipping FBL5N expansion")
            return

        today = datetime(2026, 3, 6)
        aging_buckets = [
            (0, 30, "0-30 days", 8),        # 0-30 days: 8 items
            (31, 60, "31-60 days", 8),     # 31-60 days: 8 items
            (61, 90, "61-90 days", 8),     # 61-90 days: 8 items
            (91, 365, "90+ days", 6)       # 90+ days: 6 items
        ]

        doc_num = 6000000000
        inserted = 0

        for days_min, days_max, bucket_name, count in aging_buckets:
            for i in range(count):
                doc_num += 1

                # Random aging within bucket
                days_ago = self.random.randint(days_min, days_max)
                item_date = (today - timedelta(days=days_ago)).strftime("%Y%m%d")

                customer = self.random.choice(self.customers)
                amount = self.random_amount()

                # Mix of document types
                doc_type = self.random.choice(['DR', 'DG', 'DZ'])  # Debit, Credit Memo, Payment
                shkzg = 'S' if doc_type in ['DR', 'DZ'] else 'H'  # S=debit, H=credit

                record = {
                    'MANDT': '100',
                    'BUKRS': '1000',
                    'KUNNR': customer,
                    'GJAHR': '2026',
                    'BELNR': str(doc_num),
                    'BUZEI': '001',
                    'BUDAT': item_date,
                    'BLDAT': item_date,
                    'DMBTR': str(amount),
                    'WRBTR': str(amount),
                    'WAERS': 'USD',
                    'SHKZG': shkzg,
                    'BLART': doc_type,
                    'AUGDT': '',  # Empty = open item
                    'AUGBL': '',
                    'ZFBDT': '',
                    'PRCTR': '',
                    'KOSTL': self.random.choice(self.cost_centers) if i % 2 == 0 else '',
                }

                if self._insert_record('BSID', record):
                    inserted += 1

        print(f"  Inserted {inserted} open customer items across aging buckets")

    def expand_ksb1_data(self):
        """
        Expand KSB1 - Cost Center Actual Line Items report data.
        Create actual (VERSN=000) and plan (VERSN=001) line items for CO module.
        """
        print("\n=== Expanding KSB1 - Cost Center Actual Line Items ===")

        if not self.cost_centers or not self.materials:
            print("  WARNING: Missing cost centers or materials, skipping KSB1 expansion")
            return

        # Create cost elements if they don't exist (CSKB table)
        cost_elements = [
            ('100000', 'Personnel Costs'),
            ('200000', 'Supplies'),
            ('300000', 'Depreciation'),
            ('400000', 'Services'),
            ('500000', 'Utilities'),
            ('600000', 'Maintenance'),
        ]

        # Add cost elements to CSKB
        for kstar, text in cost_elements:
            self._insert_record('CSKB', {
                'KOKRS': '1000',
                'KSTAR': kstar,
                'KTOPL': 'INT',
                'LTEXT': text,
                'MANDT': '100'
            })

        # Create actual line items (VERSN = '000')
        doc_num = 200000
        inserted_actuals = 0

        for period in ['01', '02', '03']:  # Q1 2026
            for cost_center in self.cost_centers:
                for kstar, _ in cost_elements[:3]:  # Top 3 cost elements per center
                    doc_num += 1

                    record = {
                        'KOKRS': '1000',
                        'BELNR': str(doc_num),
                        'BUZEI': '001',
                        'GJAHR': '2026',
                        'PERIO': period,
                        'VERSN': '000',  # Actual
                        'KOSTL': cost_center,
                        'KSTAR': kstar,
                        'WKG001': str(self.random_amount(10000, 100000)),
                        'WAERS': 'USD',
                        'MANDT': '100'
                    }

                    if self._insert_record('COEP', record):
                        inserted_actuals += 1

        # Create plan line items (VERSN = '001')
        inserted_plans = 0

        for period in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            for cost_center in self.cost_centers:
                for kstar, _ in cost_elements[:2]:
                    record = {
                        'KOKRS': '1000',
                        'KOSTL': cost_center,
                        'GJAHR': '2026',
                        'PERIO': period,
                        'VERSN': '001',  # Plan
                        'KSTAR': kstar,
                        'WKG001': str(self.random_amount(15000, 120000)),
                        'WAERS': 'USD',
                        'MANDT': '100'
                    }

                    if self._insert_record('COSP', record):
                        inserted_plans += 1

        print(f"  Inserted {inserted_actuals} actual CO line items (VERSN=000)")
        print(f"  Inserted {inserted_plans} plan CO line items (VERSN=001)")

    def expand_iw49n_data(self):
        """
        Expand IW49N - Cancelled Operation Report data.
        Create work orders with operations, some marked as cancelled.
        """
        print("\n=== Expanding IW49N - Cancelled Operation Report ===")

        if not self.cost_centers:
            print("  WARNING: No cost centers found, skipping IW49N expansion")
            return

        # Create work orders in AUFK
        wo_count = 0
        operation_count = 0
        cancelled_count = 0

        for i in range(15):
            self.wo_counter += 1
            aufnr = str(self.wo_counter)

            # Assign work order to cost center
            kostl = self.random.choice(self.cost_centers)
            erdat = self.random_date(days_ago_min=30, days_ago_max=120)

            # Mark some as cancelled (LOEKZ = 'X')
            loekz = 'X' if self.random.random() < 0.3 else ''

            record = {
                'MANDT': '100',
                'AUFNR': aufnr,
                'BUKRS': '1000',
                'AUFART': 'PM01',  # Preventive Maintenance
                'KOSTL': kostl,
                'GSTRS': erdat,
                'GSTAR': erdat if loekz == '' else self.random_date(days_ago_min=0, days_ago_max=30),
                'LOEKZ': loekz,
                'BSTNR': f'WO-{aufnr}',
            }

            if self._insert_record('AUFK', record):
                wo_count += 1

                # Create 2-4 operations per work order
                num_ops = self.random.randint(2, 4)
                for op_seq in range(1, num_ops + 1):
                    vornr = f"{op_seq:04d}"
                    aplzl = f"{op_seq:05d}"

                    # Mark operations as cancelled if work order is cancelled
                    op_loekz = loekz

                    record = {
                        'MANDT': '100',
                        'AUFPL': aufnr,
                        'APLZL': aplzl,
                        'VORNR': vornr,
                        'STEUS': 'PP01',  # Standard control key
                        'LTXA1': f'Operation {vornr}',
                        'ARBID': f'WC{op_seq:02d}',
                        'WERKS': '0001',
                        'ARBPL': f'0{op_seq}',
                        'LOEKZ': op_loekz,
                    }

                    if self._insert_record('AFVC', record):
                        operation_count += 1
                        if op_loekz == 'X':
                            cancelled_count += 1

                    # Create confirmation for operation (AFRU)
                    confirmation_record = {
                        'MANDT': '100',
                        'AUFPL': aufnr,
                        'APLZL': aplzl,
                        'RUBNR': f"{op_seq:04d}",
                        'BLDAT': self.random_date(days_ago_min=0, days_ago_max=60),
                        'RQDMNG': str(self.random_amount(1, 100)),
                        'LMNGA': str(self.random_amount(1, 100)),
                        'LOEKZ': op_loekz,
                    }

                    self._insert_record('AFRU', confirmation_record)

        print(f"  Inserted {wo_count} work orders")
        print(f"  Inserted {operation_count} operations ({cancelled_count} cancelled)")

    def expand_o2c_value_chain(self):
        """
        Create Order-to-Cash (O2C) value chain:
        Sales Order → Delivery → Billing → Customer Receivable
        """
        print("\n=== Expanding O2C Value Chain ===")

        if not self.customers or not self.materials:
            print("  WARNING: Missing customers or materials, skipping O2C")
            return

        vbfa_records = []
        o2c_count = 0

        for cycle in range(5):  # 5 complete O2C cycles
            # 1. Create Sales Order (VBAK/VBAP)
            vbeln_so = str(self.so_counter)
            self.so_counter += 1

            customer = self.random.choice(self.customers)
            order_date = self.random_date(days_ago_min=60, days_ago_max=180)

            so_record = {
                'MANDT': '100',
                'VBELN': vbeln_so,
                'ERDAT': order_date,
                'ERNAM': 'BAPI_USER',
                'KUNNR': customer,
                'VKORG': '1000',
                'VTWEG': '10',
                'SPART': '00',
                'VDATU': self.random_date(days_ago_min=0, days_ago_max=60),
                'WAERK': 'USD',
                'NETWR': str(self.random_amount(5000, 100000)),
                'LIFSK': 'A',
                'FAKSK': 'A',
                'LOEKZ': '',
            }

            self._insert_record('VBAK', so_record)

            # Create SO line items
            material = self.random.choice(self.materials)
            so_line_record = {
                'MANDT': '100',
                'VBELN': vbeln_so,
                'POSNR': '000010',
                'MATNR': material,
                'KWMENG': str(self.random.randint(1, 50)),
                'MEINS': 'PC',
                'NETPR': str(self.random_amount(100, 1000)),
                'PEINH': '1',
                'WAERK': 'USD',
                'VBSTAT': '0',
            }

            self._insert_record('VBAP', so_line_record)

            # 2. Create Delivery (LIKP/LIPS)
            vbeln_dl = str(self.delivery_counter)
            self.delivery_counter += 1

            dl_date = self.random_date(days_ago_min=10, days_ago_max=120)

            dl_record = {
                'MANDT': '100',
                'VBELN': vbeln_dl,
                'BUKRS': '1000',
                'KUNNR': customer,
                'ERDAT': dl_date,
                'LFART': 'LF',
                'WAERS': 'USD',
            }

            self._insert_record('LIKP', dl_record)

            # Create delivery line
            dl_line_record = {
                'MANDT': '100',
                'VBELN': vbeln_dl,
                'POSNR': '000010',
                'MATNR': material,
                'LFIMG': str(self.random.randint(1, 50)),
                'MEINS': 'PC',
                'WAERS': 'USD',
            }

            self._insert_record('LIPS', dl_line_record)

            # 3. Create document flow (VBFA): SO -> Delivery
            vbfa_so_dl = {
                'MANDT': '100',
                'VBELN': vbeln_so,
                'POSNR': '000010',
                'VBELV': vbeln_dl,
                'POSNV': '000010',
                'VBTYP_N': 'L',
                'RFMNG': str(self.random.randint(1, 50)),
            }

            self._insert_record('VBFA', vbfa_so_dl)
            vbfa_records.append(vbfa_so_dl)

            # 4. Create Invoice/Billing (VBRK/VBRP)
            vbeln_inv = str(self.invoice_counter)
            self.invoice_counter += 1

            inv_date = self.random_date(days_ago_min=0, days_ago_max=90)
            inv_amount = self.random_amount(5000, 100000)

            inv_record = {
                'MANDT': '100',
                'VBELN': vbeln_inv,
                'BUKRS': '1000',
                'KUNNR': customer,
                'ERDAT': inv_date,
                'VBTYP': 'M',
                'WAERS': 'USD',
                'NETWR': str(inv_amount),
            }

            self._insert_record('VBRK', inv_record)

            # Create invoice line
            inv_line_record = {
                'MANDT': '100',
                'VBELN': vbeln_inv,
                'POSNR': '000010',
                'MATNR': material,
                'FKIMG': str(self.random.randint(1, 50)),
                'MEINS': 'PC',
                'NETPR': str(self.random_amount(100, 1000)),
                'WAERS': 'USD',
            }

            self._insert_record('VBRP', inv_line_record)

            # 5. Create document flow: Delivery -> Invoice
            vbfa_dl_inv = {
                'MANDT': '100',
                'VBELN': vbeln_dl,
                'POSNR': '000010',
                'VBELV': vbeln_inv,
                'POSNV': '000010',
                'VBTYP_N': 'M',
                'RFMNG': str(self.random.randint(1, 50)),
            }

            self._insert_record('VBFA', vbfa_dl_inv)

            # 6. Create Customer Open Item (BSID) from invoice
            # Use invoice number as billing document number
            open_item_record = {
                'MANDT': '100',
                'BUKRS': '1000',
                'KUNNR': customer,
                'GJAHR': '2026',
                'BELNR': vbeln_inv,
                'BUZEI': '001',
                'BUDAT': inv_date,
                'BLDAT': inv_date,
                'DMBTR': str(inv_amount),
                'WRBTR': str(inv_amount),
                'WAERS': 'USD',
                'SHKZG': 'S',  # Debit
                'BLART': 'RV',  # Invoice
                'AUGDT': '',  # Open
                'AUGBL': '',
                'VBELN': vbeln_inv,
            }

            if self._insert_record('BSID', open_item_record):
                o2c_count += 1

        print(f"  Created {o2c_count} complete O2C cycles (SO → DL → INV → AR)")

    def expand_p2p_value_chain(self):
        """
        Create Procure-to-Pay (P2P) value chain:
        Requisition → PO → Goods Receipt → Invoice → Vendor Payable
        """
        print("\n=== Expanding P2P Value Chain ===")

        if not self.vendors or not self.materials:
            print("  WARNING: Missing vendors or materials, skipping P2P")
            return

        p2p_count = 0

        for cycle in range(5):  # 5 complete P2P cycles
            # 1. Create Purchase Requisition (EBAN)
            banfn = f"{self.random.randint(10000000, 99999999)}"
            req_date = self.random_date(days_ago_min=60, days_ago_max=150)

            req_record = {
                'MANDT': '100',
                'BANFN': banfn,
                'BNFPO': '0010',
                'BUKRS': '1000',
                'BSART': 'NB',
                'BADAT': req_date,
                'WAERS': 'USD',
                'MATNR': self.random.choice(self.materials),
                'MENGE': str(self.random.randint(5, 100)),
            }

            self._insert_record('EBAN', req_record)

            # 2. Create Purchase Order (EKKO/EKPO)
            ebeln = str(self.po_counter)
            self.po_counter += 1

            vendor = self.random.choice(self.vendors)
            po_date = self.random_date(days_ago_min=30, days_ago_max=120)

            po_record = {
                'MANDT': '100',
                'EBELN': ebeln,
                'BUKRS': '1000',
                'LIFNR': vendor,
                'BEDAT': po_date,
                'BSTYP': 'F',
                'BSART': 'NB',
                'LOEKZ': '',
                'WAERS': 'USD',
            }

            self._insert_record('EKKO', po_record)

            # Create PO line item
            material = self.random.choice(self.materials)
            po_qty = self.random.randint(5, 100)
            po_price = self.random_amount(10, 500)

            po_line_record = {
                'MANDT': '100',
                'EBELN': ebeln,
                'EBELP': '000010',
                'MATNR': material,
                'MENGE': str(po_qty),
                'MEINS': 'PC',
                'NETPR': str(po_price),
                'PEINH': '1',
                'WAERS': 'USD',
                'LOEKZ': '',
                'BANFN': banfn,  # Link to requisition
            }

            self._insert_record('EKPO', po_line_record)

            # 3. Create Goods Receipt (EKBE)
            gr_date = self.random_date(days_ago_min=10, days_ago_max=90)
            gr_qty = self.random.randint(1, po_qty)  # Partial or full receipt

            gr_record = {
                'MANDT': '100',
                'EBELN': ebeln,
                'EBELP': '000010',
                'BEWTP': 'E',  # Goods receipt (Wareneingang)
                'BESNR': '001',
                'BEDAT': gr_date,
                'MENGE': str(gr_qty),
                'BPMENGE': str(gr_qty),
                'DMBTR': str(gr_qty * po_price),
                'WAERS': 'USD',
            }

            self._insert_record('EKBE', gr_record)

            # 4. Create Invoice (RBKP/RSEG)
            inv_num = str(self.random.randint(100000000, 999999999))
            inv_date = self.random_date(days_ago_min=0, days_ago_max=60)
            inv_amount = gr_qty * po_price

            inv_record = {
                'MANDT': '100',
                'BUKRS': '1000',
                'BELNR': inv_num,
                'GJAHR': '2026',
                'XBLNR': f'INV-{inv_num}',
                'BUDAT': inv_date,
                'WAERS': 'USD',
            }

            self._insert_record('RBKP', inv_record)

            # Create invoice line item
            inv_line_record = {
                'MANDT': '100',
                'BELNR': inv_num,
                'GJAHR': '2026',
                'BUZEI': '001',
                'EBELN': ebeln,
                'EBELP': '000010',
                'MENGE': str(gr_qty),
                'BPMENGE': str(gr_qty),
                'DMBTR': str(inv_amount),
                'WAERS': 'USD',
            }

            self._insert_record('RSEG', inv_line_record)

            # 5. Create Vendor Open Item (BSIK)
            open_item_record = {
                'MANDT': '100',
                'BUKRS': '1000',
                'LIFNR': vendor,
                'GJAHR': '2026',
                'BELNR': inv_num,
                'BUZEI': '001',
                'DMBTR': str(inv_amount),
                'WAERS': 'USD',
                'BLART': 'RE',  # Invoice
                'BUDAT': inv_date,
                'BLDAT': inv_date,
                'SHKZG': 'H',  # Credit (payable)
                'AUGDT': '',  # Open
                'AUGBL': '',
            }

            if self._insert_record('BSIK', open_item_record):
                p2p_count += 1

        print(f"  Created {p2p_count} complete P2P cycles (REQ → PO → GR → INV → AP)")

    def expand_r2r_value_chain(self):
        """
        Create Record-to-Report (R2R) value chain:
        Ensure GL data supports period close and trial balance reporting.
        """
        print("\n=== Expanding R2R Value Chain ===")

        if not self.gl_accounts:
            print("  WARNING: No GL accounts found, skipping R2R")
            return

        # Create journal entries (BKPF/BSEG) with proper period assignments
        r2r_count = 0
        doc_counter = 7000000000

        for period in ['01', '02', '03']:  # Q1 2026
            for account in self.gl_accounts[:6]:  # Use first 6 GL accounts
                doc_counter += 1

                # Create document header
                posting_date = self.random_date(
                    days_ago_min=int((4 - int(period)) * 30),
                    days_ago_max=int((4 - int(period)) * 30) + 25
                )

                bkpf_record = {
                    'MANDT': '100',
                    'BUKRS': '1000',
                    'BELNR': str(doc_counter),
                    'GJAHR': '2026',
                    'BLART': 'KZ',  # General journal
                    'BLDAT': posting_date,
                    'BUDAT': posting_date,
                    'USNAM': 'BAPI_USER',
                    'TCODE': 'F-02',
                    'WAERS': 'USD',
                    'LOEKZ': '',
                }

                self._insert_record('BKPF', bkpf_record)

                # Create line items (must balance)
                amount1 = self.random_amount(5000, 50000)

                bseg1 = {
                    'MANDT': '100',
                    'BUKRS': '1000',
                    'BELNR': str(doc_counter),
                    'GJAHR': '2026',
                    'BUZEI': '001',
                    'SAKNR': account,
                    'DMBTR': str(amount1),
                    'WRBTR': str(amount1),
                    'SHKZG': 'S',  # Debit
                    'WAERS': 'USD',
                }

                self._insert_record('BSEG', bseg1)

                bseg2 = {
                    'MANDT': '100',
                    'BUKRS': '1000',
                    'BELNR': str(doc_counter),
                    'GJAHR': '2026',
                    'BUZEI': '002',
                    'SAKNR': self.random.choice(self.gl_accounts),
                    'DMBTR': str(amount1),
                    'WRBTR': str(amount1),
                    'SHKZG': 'H',  # Credit
                    'WAERS': 'USD',
                }

                self._insert_record('BSEG', bseg2)

                # Create period balance (GLT0)
                glt0_record = {
                    'MANDT': '100',
                    'BUKRS': '1000',
                    'SAKNR': account,
                    'GJAHR': '2026',
                    'POPER': period,
                    'HSLVT': str(amount1),
                    'KSLVT': str(amount1),
                }

                if self._insert_record('GLT0', glt0_record):
                    r2r_count += 1

        print(f"  Created {r2r_count} GL posting sets with period balances for R2R")

    def verify_expansion(self):
        """Verify the expansion by showing counts of key tables."""
        print("\n" + "=" * 70)
        print("DATABASE EXPANSION VERIFICATION - KEY METRICS")
        print("=" * 70)

        queries = [
            ("BSID - Open Customer Items",
             "SELECT COUNT(*) FROM BSID WHERE AUGDT IS NULL OR AUGDT = ''"),
            ("BSID - Total Customer Items",
             "SELECT COUNT(*) FROM BSID"),
            ("COEP - Actual CO Items (2026)",
             "SELECT COUNT(*) FROM COEP WHERE VERSN = '000' AND GJAHR = '2026'"),
            ("COSP - Plan CO Items (2026)",
             "SELECT COUNT(*) FROM COSP WHERE VERSN = '001' AND GJAHR = '2026'"),
            ("AFVC - Total Operations",
             "SELECT COUNT(*) FROM AFVC"),
            ("AFVC - Cancelled Operations",
             "SELECT COUNT(*) FROM AFVC WHERE LOEKZ = 'X'"),
            ("AFRU - Confirmations",
             "SELECT COUNT(*) FROM AFRU"),
            ("VBAK - Sales Orders",
             "SELECT COUNT(*) FROM VBAK"),
            ("LIKP - Deliveries",
             "SELECT COUNT(*) FROM LIKP"),
            ("VBRK - Invoices/Billing",
             "SELECT COUNT(*) FROM VBRK"),
            ("VBFA - Document Flow",
             "SELECT COUNT(*) FROM VBFA"),
            ("EKKO - Purchase Orders",
             "SELECT COUNT(*) FROM EKKO"),
            ("EKBE - Goods Receipts",
             "SELECT COUNT(*) FROM EKBE WHERE BEWTP = 'E'"),
            ("RBKP - Invoice Receipts",
             "SELECT COUNT(*) FROM RBKP"),
            ("BSIK - Open Vendor Items",
             "SELECT COUNT(*) FROM BSIK WHERE AUGDT IS NULL OR AUGDT = ''"),
            ("BKPF - Journal Entries",
             "SELECT COUNT(*) FROM BKPF"),
            ("GLT0 - Period Balances",
             "SELECT COUNT(*) FROM GLT0"),
        ]

        for label, query in queries:
            self.cursor.execute(query)
            count = self.cursor.fetchone()[0]
            print(f"{label:45s} : {count:6d}")

        print("=" * 70)

    def expand(self):
        """Execute the complete expansion process."""
        print("\n" + "=" * 70)
        print("SAP ECC TEST DATABASE EXPANDER")
        print("=" * 70)

        self.connect()
        self.load_master_data()

        # Expand data for three T-codes
        self.expand_fbl5n_data()  # FBL5N - Customer Open Items
        self.expand_ksb1_data()   # KSB1 - Cost Center Actual Line Items
        self.expand_iw49n_data()  # IW49N - Cancelled Operation Report

        # Expand data for three value chains
        self.expand_o2c_value_chain()  # Order-to-Cash
        self.expand_p2p_value_chain()  # Procure-to-Pay
        self.expand_r2r_value_chain()  # Record-to-Report

        # Verify and commit
        self.verify_expansion()
        self.conn.commit()

        print("\nDatabase expansion completed successfully!")
        print(f"Database location: {DB_PATH}")


if __name__ == '__main__':
    expander = SAPDatabaseExpander()
    expander.expand()
