# import pandas as pd
# import mysql.connector
# from mysql.connector import Error

# # Read the CSV file
# df = pd.read_csv(r'backend\mids-not-added.csv')

# # Data Cleaning and Type Conversion
# df['mid'] = df['mid'].astype(str)  # Convert 'mid' to string
# df['approval_date'] = pd.to_datetime(df['approval_date'], format='%d/%m/%Y', errors='coerce').dt.date  # Ensure date format
# df['closed_date'] = pd.to_datetime(df['closed_date'], errors='coerce').dt.date  # Ensure date format

# # Replace NaN with None for nullable string columns
# string_columns = ['corporation', 'dba', 'iso', 'iso_referral_type', 'sic_code', 'sic_description', 'agent1_name', 'agent2_name']
# df[string_columns] = df[string_columns].where(pd.notnull(df[string_columns]), 'None')

# # Truncate `sic_code` to fit the VARCHAR(4) constraint
# df['sic_code'] = df['sic_code'].astype(str).str[:4]  # Keep only the first 4 characters

# # Replace NaN with 0.00 for numeric columns
# numeric_columns = ['total_split', 'agent1_split', 'agent2_split']
# df[numeric_columns] = df[numeric_columns].fillna(0.00)

# print(df.tail(15))

# # Database connection and data insertion
# try:
#     connection = mysql.connector.connect(
#         host='localhost',
#         user='root',
#         password='hailhydra',
#         database='paydiverse'
#     )

#     if connection.is_connected():
#         cursor = connection.cursor()
        
#         # Insert query
#         insert_query = """
#         INSERT IGNORE INTO merchants (
#             mid, dba, corporation, iso, iso_referral_type, sic_code, sic_description,
#             agent1_name, agent2_name, total_split, is_active, is_referred,
#             approval_date, closed_date, agent1_split, agent2_split
#         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#         """

#         # Insert rows
#         for _, row in df.iterrows():
#             cursor.execute(insert_query, tuple(row.values[:16]))  # Use the first 16 columns only
        
#         connection.commit()
#         print(f"Data successfully inserted.")
# except Error as e:
#     print(f"Error: {e}")
# finally:
#     if connection.is_connected():
#         cursor.close()
#         connection.close()
# import bcrypt
# print(bcrypt.hashpw('victoria'.encode('utf-8'), bcrypt.gensalt()))

