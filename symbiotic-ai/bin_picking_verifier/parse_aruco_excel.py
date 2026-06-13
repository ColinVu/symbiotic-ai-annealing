"""Parse ArUco map from randomized_Object_List.xlsx."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import openpyxl


def parse_excel_to_aruco_map(excel_path: Path) -> Dict[str, Any]:
    """
    Convert Excel with columns: Item Name, Bin ID, Shelf, Row, Column
    to ArUco map JSON format.
    
    SKU format: {shelf.lower()}{row}{column}, e.g. 'c11', 'd34'
    """
    wb = openpyxl.load_workbook(str(excel_path), data_only=True)
    ws = wb.active
    
    bins: List[Dict[str, Any]] = []
    rows_iter = ws.iter_rows(values_only=True)
    
    # Skip header
    next(rows_iter)
    
    for row_data in rows_iter:
        if row_data[0] is None:
            break
        
        item_name = str(row_data[0])
        bin_id = int(row_data[1])
        shelf = str(row_data[2]).strip().lower()
        row_num = int(row_data[3])
        col_num = int(row_data[4])
        
        sku = f"{shelf}{row_num}{col_num}"
        
        bins.append({
            "marker_id": bin_id,
            "sku": sku,
            "row": row_num,
            "col": col_num,
            "item_name": item_name,
        })
    
    return {
        "dictionary": "DICT_5X5_1000",
        "grid": {"rows": 4, "cols": 6},
        "bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert randomized_Object_List.xlsx to ArUco map JSON"
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("../randomized_Object_List.xlsx"),
        help="Path to Excel file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aruco_map_from_excel.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()
    
    aruco_map = parse_excel_to_aruco_map(args.excel)
    
    args.output.write_text(json.dumps(aruco_map, indent=2), encoding="utf-8")
    print(f"Wrote {len(aruco_map['bins'])} bins to {args.output}")
    print(f"Sample SKUs: {[b['sku'] for b in aruco_map['bins'][:5]]}")


if __name__ == "__main__":
    main()
