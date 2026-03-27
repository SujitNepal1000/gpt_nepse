import psycopg2

passwords = ["postgres", "admin", "password", "admin123", "root", "root123", "", "nepse"]
success = False

for pwd in passwords:
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname="nepsegpt",
            user="postgres",
            password=pwd
        )
        print(f"SUCCESS with password: {pwd}")
        conn.close()
        success = True
        break
    except Exception as e:
        if "database" in str(e) and "does not exist" in str(e):
            print(f"FAILED (DB DOES NOT EXIST): {e}")
            success = True
            break
        pass

if not success:
    print("FAILED ALL")
