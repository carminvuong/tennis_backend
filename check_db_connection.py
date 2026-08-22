from db import get_connection

if __name__ == "__main__":
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        print(cur.fetchone()[0])

        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public';
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"Tables in public schema: {tables or '(none)'}")
    conn.close()

    
    print("Connected successfully.")
