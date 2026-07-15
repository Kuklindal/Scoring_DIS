import pandas as pd

from scoring.pipeline import calculate_scores

INPUT_FILE = "data/examples/ДИС.xlsx"
OUTPUT_FILE = "result.xlsx"

df = pd.read_excel(INPUT_FILE)

result_df = calculate_scores(df)

print(result_df.head())
print(result_df.columns)

result_df.to_excel(OUTPUT_FILE, index=False)

print(f"Готово. Результат сохранен в {OUTPUT_FILE}")