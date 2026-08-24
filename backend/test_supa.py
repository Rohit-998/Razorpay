import os
from supabase import create_client

url = os.environ.get("SUPABASE_URL", "https://pddwcericxdkmgpdewkw.supabase.co")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

pid = "ed0db7a4-d0ae-48ed-a57f-aac20a5af28c"

print("1. Listing 1 payment to check ID...")
list_res = supabase.table("payments").select("payment_id").limit(1).execute()
print(list_res.data)

print(f"2. Fetching {pid}...")
detail_res = supabase.table("payments").select("*").eq("payment_id", pid).execute()
print(detail_res.data)
