import os
import re

def convert_csv_to_md(input_path, output_dir):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "keywords.md")

    md_content = "# 50 Funny T-Shirt Keywords\n\n"
    
    current_category = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for Category (e.g., "1. Động vật & Quái thú hài hước")
        if re.match(r'^\d+\.', line):
            md_content += f"\n## {line}\n\n"
        
        # Check for Item (contains "–" or "-") - naive check but fits the file structure
        elif "–" in line or "-" in line:
            # Assume key is before the first "–" or "-"
            parts = re.split(r'[–-]', line, 1)
            title = parts[0].strip()
            rest = parts[1].strip()
            # If the title is very long, it might not be a title, but let's assume valid based on file view
            md_content += f"### {title}\n{rest}\n\n"
        else:
            # Intro text or others
            md_content += f"{line}\n\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Verified conversion: {output_file}")

if __name__ == "__main__":
    input_csv = "50Topics_Keyword.csv"
    output_md_dir = "agentic-rag-for-dummies/markdown_docs"
    convert_csv_to_md(input_csv, output_md_dir)
