import os, json, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
from supabase import create_client

db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

for t in ["payments", "recovery_sessions", "audit_events"]:
    res = db.table(t).select("*").limit(2).execute()
    print(f"\n=== {t} ({len(res.data)} sample rows) ===")
    if res.data:
        print("Columns:", list(res.data[0].keys()))
        for row in res.data:
            print(json.dumps(row, ensure_ascii=True, default=str)[:500])
    else:
        print("(empty)")

# Also count totals
for t in ["payments", "recovery_sessions", "audit_events"]:
    res = db.table(t).select("id", count="exact").execute()
    print(f"\n{t} total count: {res.count}")
