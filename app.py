import math
import bcrypt
from flask import Flask, jsonify, request, render_template
from flask_mail import Mail, Message
from flask_mysqldb import MySQL
from flask_cors import CORS
import jwt
from functools import wraps
from dotenv import load_dotenv
from scripts import extract_iso_and_date, process_uploaded_file, process_authorize_file, process_nuvei_file, process_nmi_file, process_cwa_file, process_rac_file, process_paymentcloud_file, process_highrisk_file, process_seamless_file, process_payscout_file, process_micamp_file, process_merchant_industry_file, process_ccbill_file, process_pepper_pay_file, process_quantum_file, process_seamless_paynote_file
import os
from decimal import Decimal
import pandas as pd
import random
import string
import datetime

app = Flask(__name__)
CORS(app)

load_dotenv()

# Configure MySQL connection details
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')  # Replace with your SMTP server
app.config['MAIL_PORT'] = os.getenv('MAIL_PORT')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS')
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  # Your email
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # Your email password or app-specific password
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

# Initialize MySQL
mysql = MySQL(app)
mail = Mail(app)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # Check if the token is provided in the request headers
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]  # Get the token part

        if not token:
            return jsonify({'message': 'Unauthorized'}), 401  # Unauthorized

        try:
            # Decode token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['id']

            cur = mysql.connection.cursor()
            cur.execute('SELECT deleted_at FROM users WHERE id = %s', (current_user,))
            user = cur.fetchone()
            cur.close()

            if not user or user[0] is not None:  # If user is deleted
                return jsonify({'message': 'Unauthorized. User does not exist.'}), 401

        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401
        except Exception as e:
            return jsonify({'message': 'An error occurred', 'error': str(e)}), 500

        return f(current_user, *args, **kwargs)

    return decorated

@app.route('/merchants')
@token_required  # Apply the token validation decorator
def index(current_user):
    cur = mysql.connection.cursor()
    cur.execute('SELECT agent1_name, agent1_split, agent2_name, agent2_split, approval_date, closed_date, corporation, dba, is_active, is_referred, iso, iso_referral_type, mid FROM merchants')
    column_names = [i[0] for i in cur.description]
    data = cur.fetchall()
    cur.close()
    result = [dict(zip(column_names, row)) for row in data]  # Convert rows to dict
    return jsonify(result)  # Return JSON response


@app.route('/unique-merchants')
@token_required
def get_unique_corporations(current_user):
    cur = mysql.connection.cursor()
    cur.execute('SELECT DISTINCT corporation FROM merchants WHERE corporation IS NOT NULL AND corporation != ""')
    column_names = [i[0] for i in cur.description]
    data = cur.fetchall()
    cur.close()
    result = [dict(zip(column_names, row)) for row in data]
    
    return jsonify(result)


@app.route('/volume-per-month', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_volume_per_month(current_user):
    date = request.args.get('date')  # Get the date from query parameters
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()  # Regular cursor
        query = '''
            SELECT SUM(VOLUME) AS total_volume 
            FROM revenue 
            WHERE DATE(date) = DATE(%s)
        '''
        cur.execute(query, (date,))
        result = cur.fetchone()
        cur.close()

        if result and result[0] is not None:
            return jsonify({"total_volume": float(result[0])}), 200
        else:
            return jsonify({"total_volume": 0}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/revenue-per-month', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_revenue_per_month(current_user):
    date = request.args.get('date')  # Get the date from query parameters
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()  # Regular cursor
        query = '''
            SELECT SUM(total_revenue) AS total_revenue
            FROM (
                SELECT SUM(paydiverse_residual) AS total_revenue 
                FROM revenue 
                WHERE DATE(date) = DATE(%s)

                UNION ALL

                SELECT SUM(adjustment_price) AS total_revenue 
                FROM adjustments 
                WHERE DATE(date) = DATE(%s)
            ) AS combined;
        '''
        cur.execute(query, (date, date,))
        result = cur.fetchone()
        cur.close()

        if result and result[0] is not None:
            return jsonify({"total_revenue": float(result[0])}), 200
        else:
            return jsonify({"total_revenue": 0}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/active-mids', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_active_mids(current_user):
    date = request.args.get('date')  # Get the date from query parameters
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()  # Regular cursor
        query = '''
            SELECT COUNT(*) AS total_mids 
            FROM revenue as r
            INNER JOIN iso as i on r.iso = i.iso 
            WHERE date = %s AND paydiverse_residual != 0 AND i.referral_type = 'MID' 
        '''
        cur.execute(query, (date,))
        result = cur.fetchone()
        cur.close()

        if result and result[0] is not None:
            return jsonify({"total_mids": float(result[0])}), 200
        else:
            return jsonify({"total_mids": 0}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/referral-type-residual', methods=['GET'])
@token_required
def get_referral_type_residual(current_user):
    date = request.args.get('date')
    
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    # Define all possible referral types
    all_referral_types = ['3rd Party', 'MID', 'Gateway']

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT I.referral_type AS 'Referral Type', SUM(R.paydiverse_residual) AS 'Total Residual'
        FROM revenue AS R
        INNER JOIN iso as I ON R.iso = I.iso
        GROUP BY I.referral_type, R.date
        HAVING R.date = %s
        '''
        cur.execute(query, (date,))
        results = cur.fetchall()
        cur.close()

        result_dict = {row[0]: float(row[1]) for row in results}

        data = [
            {
                'referral_type': ref_type,
                'total_residual': result_dict.get(ref_type, 0.0)
            }
            for ref_type in all_referral_types
        ]

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/revenue-per-mid', methods=['GET'])
@token_required
def get_revenue_per_mid(current_user):
    date = request.args.get('date')
    
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT M.dba, R.mid, R.iso, R.total_residual, R.paydiverse_residual, M.corporation AS corporation, R.agent1_name AS agent1_name, R.agent1_percentage AS agent1_percentage, R.agent1_payout AS agent1_payout, R.agent2_name AS agent2_name, R.agent2_percentage AS agent2_percentage, R.agent2_payout AS agent2_payout
        FROM revenue AS R
        INNER JOIN merchants AS M
        ON R.mid = M.mid
        JOIN iso as I ON R.iso = I.iso
        WHERE date = %s and R.paydiverse_residual != 0 AND I.referral_type = 'MID'
        ORDER BY 5 DESC
        '''
        cur.execute(query, (date,))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'dba': row[0],
                'mid': row[1],
                'iso': row[2],
                'total_residual': float(row[3]),
                'paydiverse_residual': float(row[4]),
                'corporation': row[5],
                'agent1_name': row[6],
                'agent1_percentage': row[7],
                'agent1_payout': row[8],
                'agent2_name': row[9],
                'agent2_percentage': row[10],
                'agent2_payout': row[11],
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/revenue-per-corporation', methods=['GET'])
@token_required
def get_revenue_per_corporation(current_user):
    date = request.args.get('date')
    
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT M.corporation, SUM(R.paydiverse_residual)
        FROM revenue AS R
        INNER JOIN merchants AS M
        ON R.mid = M.mid
        WHERE date = %s 
        GROUP BY M.corporation
        ORDER BY 2 DESC
        '''
        cur.execute(query, (date,))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'corporation': row[0],
                'paydiverse_residual': float(row[1])
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/negative-revenue-per-corporation', methods=['GET'])
@token_required
def get_negative_revenue_per_corporation(current_user):
    date = request.args.get('date')
    
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT M.dba, R.mid, R.iso, R.total_residual, R.paydiverse_residual, M.corporation
        FROM revenue AS R
        INNER JOIN merchants AS M
        ON R.mid = M.mid
        WHERE date = %s AND R.paydiverse_residual < 0
        ORDER BY 5 DESC
        '''
        cur.execute(query, (date,))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'dba': row[0],
                'mid': row[1],
                'iso': row[2],
                'total_residual': float(row[3]),
                'paydiverse_residual': float(row[4]),
                'corporation': row[5]
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/revenue-per-industry', methods=['GET'])
@token_required
def get_revenue_per_industry(current_user):
    date = request.args.get('date')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 100))  # Default to 100 items per page
    
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()
        
        # Query to get total count
        count_query = '''
        SELECT COUNT(*) 
        FROM (
            SELECT S.four_digit_sic_codes
            FROM revenue as R
            INNER JOIN merchants as M ON M.mid = R.mid
            INNER JOIN sic_codes as S ON M.sic_code = S.four_digit_sic_codes
            WHERE date = %s
            GROUP BY S.four_digit_sic_codes
        ) as subquery
        '''
        cur.execute(count_query, (date,))
        total_count = cur.fetchone()[0]

        # Main query with pagination
        query = '''
        SELECT S.four_digit_sic_code_descriptions, SUM(R.total_residual) AS 'total_residual', SUM(R.paydiverse_residual) AS 'paydiverse_residual'
        FROM revenue as R
        INNER JOIN merchants as M ON M.mid = R.mid
        INNER JOIN sic_codes as S ON M.sic_code = S.four_digit_sic_codes
        WHERE date = %s
        GROUP BY S.four_digit_sic_codes
        ORDER BY SUM(R.paydiverse_residual) DESC
        LIMIT %s OFFSET %s
        '''
        offset = (page - 1) * page_size
        cur.execute(query, (date, page_size, offset))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'four_digit_sic_code_descriptions': row[0],
                'total_residual': float(row[1]),
                'paydiverse_residual': float(row[2])
            } for row in results]
            
            return jsonify({
                'data': data,
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': math.ceil(total_count / page_size)
            }), 200
        else:
            return jsonify({
                'data': [],
                'page': page,
                'page_size': page_size,
                'total_count': 0,
                'total_pages': 0
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/revenue-per-agent', methods=['GET'])
@token_required
def get_revenue_per_agent(current_user):
    date = request.args.get('date')
    
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT mid, iso, agent1_name, agent2_name, total_residual, paydiverse_residual, agent1_payout, agent2_payout
        FROM revenue
        WHERE agent1_name IS NOT NULL AND date = %s
        '''
        cur.execute(query, (date,))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'mid': row[0],
                'iso': row[1],
                'agent1_name': row[2],
                'agent2_name': row[3],
                'total_residual': float(row[4]),
                'paydiverse_residual': float(row[5]),
                'agent1_payout': row[6],
                'agent2_payout': row[7]
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/agents-payout', methods=['GET'])
@token_required
def get_agents_payout(current_user):
    date = request.args.get('date')
    
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT agent_name, SUM(payout) AS total_payout
        FROM (
            SELECT agent1_name AS agent_name, agent1_payout AS payout
            FROM revenue
            WHERE agent1_name IS NOT NULL AND date = %s
            UNION ALL
            SELECT agent2_name AS agent_name, agent2_payout AS payout
            FROM revenue
            WHERE agent2_name IS NOT NULL AND date = %s
        ) AS combined_agents
        GROUP BY agent_name
        ORDER BY 2 DESC;
        '''
        cur.execute(query, (date, date))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'agent_name': row[0],
                'total_payout': float(row[1])
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/unique-mid-dba', methods=['GET'])
@token_required
def get_unique_mid_dba(current_user):
    cur = mysql.connection.cursor()
    cur.execute('SELECT DISTINCT mid, dba, iso FROM merchants')
    column_names = [i[0] for i in cur.description]
    data = cur.fetchall()
    cur.close()
    result = [dict(zip(column_names, row)) for row in data]
    
    return jsonify(result)


@app.route('/unique-iso', methods=['GET'])
@token_required
def get_unique_iso(current_user):
    cur = mysql.connection.cursor()
    cur.execute('SELECT DISTINCT id, iso, is_active FROM iso')
    column_names = [i[0] for i in cur.description]
    data = cur.fetchall()
    cur.close()
    result = [dict(zip(column_names, row)) for row in data]
    
    return jsonify(result)


@app.route('/unique-agent-name', methods=['GET'])
@token_required
def get_unique_agent_names(current_user):
    cur = mysql.connection.cursor()
    cur.execute('''
                SELECT agent1_name AS agent_name
                FROM revenue
                WHERE agent1_name IS NOT NULL

                UNION

                SELECT agent2_name AS agent_name
                FROM revenue
                WHERE agent2_name IS NOT NULL;
                ''')
    column_names = [i[0] for i in cur.description]
    data = cur.fetchall()
    cur.close()
    result = [dict(zip(column_names, row)) for row in data]
    
    return jsonify(result)


@app.route('/mid-insights', methods=['GET'])
@token_required
def get_mid_insights(current_user):
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    mid = request.args.get('mid')
    
    if not start_date or not end_date or not mid:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        WITH RECURSIVE months AS (
            SELECT DATE(%s) AS month
            UNION ALL
            SELECT DATE_ADD(month, INTERVAL 1 MONTH)
            FROM months
            WHERE month < %s
        )
        SELECT 
            m.month, 
            COALESCE(SUM(r.total_residual), 0) AS total_residual, 
            COALESCE(SUM(r.paydiverse_residual), 0) AS paydiverse_residual
        FROM months m
        LEFT JOIN revenue r ON m.month = r.DATE AND r.mid = %s
        GROUP BY m.month
        ORDER BY m.month ASC;
        '''
        cur.execute(query, (start_date, end_date, mid))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'month': row[0],
                'total_residual': float(row[1]),
                'paydiverse_residual': float(row[2])
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/agent-insights', methods=['GET'])
@token_required
def get_agent_insights(current_user):
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    agent_name = request.args.get('agent_name')
    
    if not start_date or not end_date or not agent_name:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT date AS month, SUM(total_residual) AS total_residual, SUM(paydiverse_residual) AS paydiverse_residual
        FROM revenue
        WHERE (agent1_name = %s OR agent2_name = %s) AND (date between %s AND %s)
        GROUP BY date;
        '''
        cur.execute(query, (agent_name, agent_name, start_date, end_date))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'month': row[0],
                'total_residual': float(row[1]),
                'paydiverse_residual': float(row[2])
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/iso-insights', methods=['GET'])
@token_required
def get_iso_insights(current_user):
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    iso = request.args.get('iso')
    
    if not start_date or not end_date or not iso:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT date AS month, SUM(total_residual) AS total_residual, SUM(paydiverse_residual) AS paydiverse_residual
        FROM revenue
        WHERE (iso = %s) AND (date between %s AND %s)
        GROUP BY date;
        '''
        cur.execute(query, (iso, start_date, end_date))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'month': row[0],
                'total_residual': float(row[1]),
                'paydiverse_residual': float(row[2])
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/agents-mid-list', methods=['GET'])
@token_required
def get_agents_mid_list(current_user):
    agent_name = request.args.get('agent_name')
    
    if not agent_name:
        return jsonify({"error": "Agent Name is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT mid, dba, iso
        FROM merchants
        WHERE agent1_name = %s OR agent2_name = %s;
        '''
        cur.execute(query, (agent_name, agent_name))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'mid': row[0],
                'dba': row[1],
                'iso': row[2]
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/iso-per-month', methods=['GET'])
@token_required
def get_iso_per_month(current_user):
    date = request.args.get('date')
    
    if not date:
        return jsonify({"error": "Date is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT 
            i.id,  -- Include i.id to ensure no ambiguity
            i.iso, 
            SUM(r.total_residual) AS total_residual, 
            SUM(r.paydiverse_residual) + 
            COALESCE(
                (SELECT SUM(a.adjustment_price) 
                FROM adjustments a 
                WHERE i.id = a.iso_id AND r.date = a.date), 0
            ) AS paydiverse_residual,
            COALESCE(MAX(a.adjustment_price), 0) AS adjustment_price
        FROM revenue r
        LEFT JOIN iso i ON r.iso = i.iso
        LEFT JOIN adjustments a ON i.id = a.iso_id AND r.date = a.date
        WHERE r.date = %s
        AND i.iso IS NOT NULL 
        GROUP BY i.id, i.iso  -- Ensure i.id is included
        ORDER BY 1;
        '''
        cur.execute(query, (date, ))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'id': float(row[0]),
                'iso': row[1],
                'total_residual': float(row[2]),
                'paydiverse_residual': float(row[3]),
                'adjustment_price': float(row[4])
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/each-iso-data', methods=['GET'])
@token_required
def each_iso_data(current_user):
    date = request.args.get('date')
    iso = request.args.get('iso')
    
    if not date and not iso:
        return jsonify({"error": "Date and ISO are required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT r.mid, m.corporation, m.dba, total_residual, paydiverse_residual, r.agent1_name, r.agent2_name, r.agent1_payout, r.agent2_payout, r.agent1_percentage, r.agent2_percentage
        FROM revenue AS r
        INNER JOIN merchants AS m ON r.mid = m.mid
        WHERE r.iso = %s AND date = %s
        '''
        cur.execute(query, (iso, date, ))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'mid': row[0],
                'corporation': row[1],
                'dba': row[2],
                'total_residual': float(row[3]),
                'paydiverse_residual': float(row[4]),
                'agent1_name': row[5],
                'agent2_name': row[6],
                'agent1_payout': row[7],
                'agent2_payout': row[8],
                'agent1_percentage': row[9],
                'agent2_percentage': row[10],
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

@app.route('/each-corp-data', methods=['GET'])
@token_required
def each_corp_data(current_user):
    date = request.args.get('date')
    corporation = request.args.get('corporation')
    
    if not date and not corporation:
        return jsonify({"error": "Date and Corporation are required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT r.mid, r.iso, m.dba, paydiverse_residual, total_residual
        FROM merchants as m
        INNER JOIN revenue as r
        ON r.mid = m.mid 
        WHERE corporation = %s AND date = %s
        '''
        cur.execute(query, (corporation, date, ))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'dba': row[2],
                'iso': row[1],
                'mid': row[0],
                'total_residual': float(row[4]),
                'paydiverse_residual': float(row[3])
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/each-agent-data', methods=['GET'])
@token_required
def each_agent_data(current_user):
    date = request.args.get('date')
    agent_name = request.args.get('agent_name')
    
    if not date and not agent_name:
        return jsonify({"error": "Date and Agent name are required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT 
            r.mid AS mid,
            r.iso AS iso,
            r.dba AS dba,
            m.corporation AS corporation,
            r.total_residual AS total_residual,
            r.paydiverse_residual AS paydiverse_residual,
            CASE 
                WHEN r.agent1_name = %s THEN r.agent1_percentage
                WHEN r.agent2_name = %s THEN r.agent2_percentage
            END AS agent_percentage,
            CASE 
                WHEN r.agent1_name = %s THEN r.agent1_payout
                WHEN r.agent2_name = %s THEN r.agent2_payout
            END AS agent_payout
        FROM 
            revenue AS r
        INNER JOIN 
            merchants AS m ON r.mid = m.mid
        WHERE 
            (r.agent1_name = %s OR r.agent2_name = %s)
            AND date = %s;
        '''
        cur.execute(query, (agent_name, agent_name, agent_name, agent_name, agent_name, agent_name, date, ))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'mid': row[0],
                'iso': row[1],
                'dba': row[2],
                'corporation': row[3],
                'total_residual': float(row[4]),
                'paydiverse_residual': float(row[5]),
                'agent_percentage': row[6],
                'agent_payout': float(row[7]),
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/revenue-per-month-each-iso', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_revenue_per_month_each_iso(current_user):
    start_date = request.args.get('start_date')  # Get the date from query parameters
    end_date = request.args.get('end_date')  # Get the date from query parameters
    iso = request.args.get('iso')    # Get the optional iso parameter

    if not start_date and end_date:
        return jsonify({"error": "Start date and end date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()

        if iso != "":
            # Query for a specific ISO
            query = '''
                SELECT SUM(paydiverse_residual) AS total_revenue, date 
                FROM revenue 
                WHERE iso = %s AND date between %s and %s
                GROUP BY date
            '''
            cur.execute(query, (iso, start_date, end_date, ))
        else:
            # Query for all ISOs
            query = '''
                SELECT SUM(paydiverse_residual) AS total_revenue, date 
                FROM revenue 
                WHERE date between %s and %s
                GROUP BY date
            '''
            cur.execute(query, (start_date, end_date, ))

        results = cur.fetchall()
        cur.close()

        # Format response
        data = [{"date": row[1], "total_revenue": float(row[0])} for row in results]

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get-logs', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_logs(current_user):
    try:
        # Get the number of days from the query parameters, default to 1 if not provided
        days = request.args.get('days', default=1, type=int)

        # Validate the `days` parameter to prevent invalid or negative values
        if days not in [1, 7, 14, 30, 45, 60, 90, 120]:
            return jsonify({"error": "Invalid 'days' parameter. Allowed values are 1, 7, 14, 30, 45, 60, 90, and 120."}), 400

        cur = mysql.connection.cursor()

        query = f'''
            SELECT * 
            FROM revenue
            WHERE updated_at >= NOW() - INTERVAL %s DAY
            ORDER BY updated_at DESC;
        '''
        cur.execute(query, (days,))

        results = cur.fetchall()
        cur.close()

        # Format response
        data = [{
            "date": row[0], 
            "iso": row[1], 
            "mid": row[2], 
            "dba": row[3], 
            "volume": float(row[4]),
            "total_residual": float(row[5]),
            "paydiverse_residual": float(row[6]),
            "agent1_name": row[7],
            "agent1_percentage": row[8],
            "agent1_payout": row[9],
            "agent2_name": row[10],
            "agent2_percentage": row[11],
            "agent2_payout": row[12],
        } for row in results]

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/add-revenue', methods=['POST'])
@token_required
def add_revenue(current_user):
    """
    Endpoint to insert revenue data into the revenue table.
    Accepts date, iso, and paydiverse_residual from the frontend.
    Other values are set as specified in the request.
    """
    try:
        # Parse input JSON
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request. No data provided."}), 400

        # Extract required fields
        date = data.get('date')
        iso = data.get('iso')
        paydiverse_residual = data.get('paydiverse_residual')

        # Validate required fields
        if not date or not iso or paydiverse_residual is None:
            return jsonify({"error": "Missing required fields: date, iso, paydiverse_residual"}), 400

        # Check if the revenue for the given date and ISO already exists
        cur = mysql.connection.cursor()
        iso_concatenated = iso + '%'  # Adjusting ISO for the query pattern
        check_query = '''
            SELECT COUNT(*) FROM revenue
            WHERE iso LIKE %s AND date = %s
        '''
        cur.execute(check_query, (iso_concatenated, date))
        if cur.fetchone()[0] > 0:
            cur.close()
            return jsonify({"message": "Revenue has already been uploaded!"}), 200

        # Hardcoded values
        mid = '111123456'
        dba = iso
        total_residual = paydiverse_residual
        volume = 0

        # Insert into the revenue table
        insert_query = '''
            INSERT INTO revenue (date, iso, mid, dba, volume, total_residual, paydiverse_residual, updated_by,
                                 agent1_name, agent1_percentage, agent1_payout,
                                 agent2_name, agent2_percentage, agent2_payout)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, NULL, NULL, NULL, NULL)
        '''
        cur.execute(insert_query, (date, iso, mid, dba, volume, total_residual, paydiverse_residual, current_user))
        mysql.connection.commit()
        cur.close()

        # Success response
        return jsonify({"message": "Revenue data successfully added!"}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/import-data', methods=['POST'])
@token_required
def import_data(current_user):
    # Step 1: Validate and process the uploaded file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file!"}), 400

    # Extract filename without extension
    filename = file.filename.rsplit('.', 1)[0]

    # Step 2: Extract ISO and date from the filename
    extraction_result = extract_iso_and_date(filename)

    if "error" in extraction_result:
        return jsonify(extraction_result), 400

    iso = extraction_result["iso"]
    iso_concatenated = iso + '%'
    date = extraction_result["date"]
    
    try:
        # Step 3: Check if the file has already been processed
        cur = mysql.connection.cursor()
        query = '''
            SELECT COUNT(*) FROM revenue
            WHERE iso LIKE %s AND date = %s
        '''
        cur.execute(query, (iso_concatenated, date))
        if cur.fetchone()[0] > 0:
            return jsonify({"message": "File has already been uploaded!"}), 200

        # Step 4: Process the uploaded file based on ISO
        mapping_file = os.path.join(os.path.dirname(__file__), 'column_mapping.json')
        if iso == 'Authorize.Net':
            extracted_data = process_authorize_file(file)
        elif iso == 'Nuvei':
            extracted_data = process_nuvei_file(file)
        elif iso == 'NMI':
            extracted_data = process_nmi_file(file)
        elif iso == 'Cardworks':
            extracted_data = process_cwa_file(file)
        elif iso == 'RAC':
            extracted_data = process_rac_file(file)
        elif "Payment Cloud" in iso:
            extracted_data = process_paymentcloud_file(file)
        elif "The HiRisk Processor" in iso:
            extracted_data = process_highrisk_file(file)
        elif "PayScout" in iso:
            extracted_data = process_payscout_file(file, file.filename)
        elif "Seamless Chex" == iso:
            extracted_data = process_seamless_file(file)
        elif "MIcamp NMI" in iso:
            extracted_data = process_micamp_file(file, file.filename)
        elif iso == "Merchant Industry":
            extracted_data = process_merchant_industry_file(file, file.filename)
        elif iso == "CC Bill":
            extracted_data = process_ccbill_file(file, file.filename)
        elif iso == "Pepper Pay":
            extracted_data = process_pepper_pay_file(file)
        elif iso == "Quantum":
            extracted_data = process_quantum_file(file)
        elif iso == "Seamless Chex Paynote":
            extracted_data = process_seamless_paynote_file(file)
        else:
            extracted_data = process_uploaded_file(file, mapping_file)
        mids = extracted_data['mid']
        
        # Step 5: Validate all MIDs are in the merchants table
        try:
            missing_mids = []
            for mid in mids:
                merchant_query = '''
                    SELECT COUNT(*) FROM merchants
                    WHERE mid = %s
                '''
                cur.execute(merchant_query, (mid,))
                if cur.fetchone()[0] == 0:
                    missing_mids.append(mid)
            
            if missing_mids:
                return jsonify({
                    "error": "File processing aborted. The following MIDs are not present in the merchants table:",
                    "missing_mids": missing_mids
                }), 400
        except Exception as e:
            return jsonify({"error": f"Error validating MIDs: {str(e)}"}), 500

        # Step 6: Fetch additional merchant information
        referred_data = []
        for mid in mids:
            merchant_query = '''
                SELECT dba, is_referred, agent1_name, agent1_split, agent2_name, agent2_split
                FROM merchants
                WHERE mid = %s
            '''
            cur.execute(merchant_query, (mid,))
            merchant_data = cur.fetchone()
            
            if merchant_data:
                merchant_dba, is_referred, agent1_name, agent1_split, agent2_name, agent2_split = merchant_data

                # Store referred data if applicable
                if is_referred == 1:
                    referred_data.append({
                        "mid": mid,
                        "agent1_name": agent1_name,
                        "agent1_split": agent1_split,
                        "agent2_name": agent2_name,
                        "agent2_split": agent2_split
                    })

                # Fallback for dba if missing in extracted_data
                dba_index = extracted_data[extracted_data['mid'] == mid].index

                # Check if there are matching indices and if 'dba' is NaN
                if not (dba_index.empty and pd.isna(extracted_data.loc[dba_index[0], 'dba'])):
                    extracted_data.loc[dba_index[0], 'dba'] = merchant_dba
        
        # Step 7: Insert data into the revenue table
        for i, mid in enumerate(mids):
            dba = extracted_data['dba'][i]
            volume = extracted_data['volume'][i]
            total_residual = extracted_data['total_residual'][i]
            paydiverse_residual = extracted_data['paydiverse_residual'][i]
            # Conditionally set iso for Payment Cloud
            iso_to_insert = extracted_data['iso'][i] if "Payment Cloud" in iso else iso
            # Check if the mid has referred data
            referred = next((item for item in referred_data if item['mid'] == mid), None)
            if referred:
                # Calculate agent payouts
                agent1_name = referred['agent1_name']
                agent1_split = referred['agent1_split']
                agent2_name = referred.get('agent2_name', None)
                agent2_split = referred.get('agent2_split', None)

                paydiverse_residual_decimal = Decimal(paydiverse_residual)
                agent1_payout = paydiverse_residual_decimal * (agent1_split / Decimal(100))
                agent2_payout = (
                    paydiverse_residual_decimal * (agent2_split / Decimal(100))
                    if agent2_split
                    else None
                )

                # Insert into revenue table with referral details
                insert_query = '''
                    INSERT INTO revenue (date, iso, mid, dba, volume, total_residual, paydiverse_residual,
                                         agent1_name, agent1_percentage, agent1_payout, 
                                         agent2_name, agent2_percentage, agent2_payout, updated_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                '''
                cur.execute(insert_query, (date, iso_to_insert, mid, dba, volume, total_residual, paydiverse_residual,
                                           agent1_name, agent1_split, float(agent1_payout),
                                           agent2_name, agent2_split, float(agent2_payout) if agent2_payout else None, current_user))
            else:
                # Insert without referral details
                insert_query = '''
                    INSERT INTO revenue (date, iso, mid, dba, volume, total_residual, paydiverse_residual, updated_by,
                                         agent1_name, agent1_percentage, agent1_payout,
                                         agent2_name, agent2_percentage, agent2_payout)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, NULL, NULL, NULL, NULL)
                '''
                cur.execute(insert_query, (date, iso_to_insert, mid, dba, volume, total_residual, paydiverse_residual, current_user))

        # Commit changes to the database
        mysql.connection.commit()
        cur.close()

        # Step 8: Return success response
        return jsonify({
            "message": "Data successfully imported.",
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

  

@app.route('/edit-log', methods=['POST'])
@token_required
def edit_log(current_user):
    data = request.get_json()

    # Validate required fields
    required_fields = [
        'date', 'iso', 'mid', 'old_mid', 'dba', 'volume',
        'total_residual', 'paydiverse_residual', 'agent1_name',
        'agent1_percentage', 'agent1_payout', 'agent2_name',
        'agent2_percentage', 'agent2_payout'
    ]

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'message': f'Missing fields: {", ".join(missing_fields)}'}), 400

    try:
        # Extract composite key values
        date = data['date']
        iso = data['iso']
        old_mid = data['old_mid']  # Use old_mid to locate the row

        # Extract fields to update
        update_fields = {field: data[field] for field in required_fields if field not in ['date', 'iso', 'old_mid']}
        columns = ', '.join(f"{key} = %s" for key in update_fields.keys())
        values = list(update_fields.values())

        # Construct update query
        query = f"""
            UPDATE revenue 
            SET {columns}
            WHERE date = %s AND iso = %s AND mid = %s
        """
        values.extend([date, iso, old_mid])  # Use old_mid to locate the row

        # Execute the query
        cur = mysql.connection.cursor()
        cur.execute(query, tuple(values))
        mysql.connection.commit()

        # Check if any rows were updated
        if cur.rowcount == 0:
            return jsonify({'message': 'Log entry not found or no changes made'}), 404

        cur.close()
        return jsonify({'message': 'Log updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete-log', methods=['DELETE'])
@token_required
def delete_log(current_user):
    data = request.get_json()
    required_fields = ['date', 'iso', 'mid']
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        return jsonify({'message': f'Missing fields: {", ".join(missing_fields)}'}), 400

    try:
        date = data['date']
        iso = data['iso']
        mid = data['mid']

        query = """
            DELETE FROM revenue
            WHERE date = %s AND iso = %s AND mid = %s
        """
        values = (date, iso, mid)

        cur = mysql.connection.cursor()
        cur.execute(query, values)
        mysql.connection.commit()
        cur.close()

        if cur.rowcount == 0:
            return jsonify({'message': 'Log entry not found'}), 404

        return jsonify({'message': 'Log deleted successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/merchants/<string:mid>', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_merchant_by_mid(current_user, mid):
    cur = mysql.connection.cursor()
    
    # Query to fetch merchant by MID
    cur.execute('SELECT agent1_name, agent1_split, agent2_name, agent2_split, approval_date, closed_date, corporation, dba, is_active, is_referred, iso, iso_referral_type, sic_code, sic_description, mid FROM merchants WHERE mid = %s', (mid,))
    
    # Fetch the result
    data = cur.fetchone()
    
    # Check if the merchant exists
    if data:
        column_names = [i[0] for i in cur.description]
        result = dict(zip(column_names, data))
        cur.close()
        return jsonify(result), 200  # Return JSON response with HTTP status 200
    else:
        cur.close()
        return jsonify({'message': 'Merchant not found'}), 200  # Return 404 if MID not found


@app.route('/merchants', methods=['POST'])
@token_required
def add_or_edit_merchant(current_user):
    data = request.get_json()

    # Required fields
    required_fields = [
        "mid", "iso", "dba", "corporation", "is_active",
        "is_referred", "iso_referral_type", "approval_date",
        "closed_date", "sic_code", "sic_description"
    ]

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'message': f'Missing fields: {", ".join(missing_fields)}'}), 400

    # Extract fields
    mid = data['mid']
    iso = data['iso']
    dba = data['dba']
    corporation = data['corporation']
    is_active = data['is_active']
    is_referred = data['is_referred']
    iso_referral_type = data['iso_referral_type']
    approval_date = data.get('approval_date')  # Allow null for dates
    closed_date = data.get('closed_date')
    sic_code = data.get('sic_code')
    sic_code_description = data.get('sic_description')

    # Agent fields will only be included if `is_referred` is true
    agent1_name = data.get('agent1_name') if is_referred else None
    agent1_split = data.get('agent1_split') if is_referred else None
    agent2_name = data.get('agent2_name') if is_referred else None
    agent2_split = data.get('agent2_split') if is_referred else None

    try:
        cur = mysql.connection.cursor()

        # Check if the MID exists
        cur.execute('SELECT COUNT(*) FROM merchants WHERE mid = %s', (mid,))
        exists = cur.fetchone()[0] > 0

        if exists:
            # Update existing merchant
            query = """
                UPDATE merchants
                SET iso = %s, dba = %s, corporation = %s, is_active = %s,
                    is_referred = %s, iso_referral_type = %s, approval_date = %s,
                    closed_date = %s, agent1_name = %s, agent1_split = %s,
                    agent2_name = %s, agent2_split = %s, sic_code = %s,
                    sic_description = %s, updated_by = %s
                WHERE mid = %s
            """
            values = (
                iso, dba, corporation, is_active, is_referred, iso_referral_type,
                approval_date, closed_date, agent1_name, agent1_split,
                agent2_name, agent2_split, sic_code, sic_code_description, current_user, mid
            )
            cur.execute(query, values)
            message = "Merchant updated successfully."
        else:
            # Insert new merchant
            query = """
                INSERT INTO merchants (mid, iso, dba, corporation, is_active, 
                    is_referred, iso_referral_type, approval_date, closed_date, 
                    agent1_name, agent1_split, agent2_name, agent2_split, sic_code, 
                    sic_description, updated_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                mid, iso, dba, corporation, is_active, is_referred, iso_referral_type,
                approval_date, closed_date, agent1_name, agent1_split, agent2_name,
                agent2_split, sic_code, sic_code_description, current_user
            )
            cur.execute(query, values)
            message = "Merchant added successfully."

        mysql.connection.commit()
        cur.close()
        return jsonify({'message': message}), 200

    except Exception as e:
        return jsonify({'message': 'An error occurred.', 'error': str(e)}), 500
    

@app.route('/merchants/<string:mid>', methods=['DELETE'])
@token_required
def delete_merchant(current_user, mid):
    try:
        cur = mysql.connection.cursor()

        # Check if the MID exists in the database
        cur.execute('SELECT COUNT(*) FROM merchants WHERE mid = %s', (mid,))
        exists = cur.fetchone()[0] > 0

        if not exists:
            return jsonify({'message': 'Merchant not found'}), 404

        # Delete the merchant with the specified MID
        cur.execute('DELETE FROM merchants WHERE mid = %s', (mid,))
        mysql.connection.commit()
        cur.close()

        return jsonify({'message': 'Merchant deleted successfully'}), 200
    except Exception as e:
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500


def send_welcome_email(email, full_name, password):
    msg_title = 'Welcome to Victoria!'
    msg_body = f'Dear {full_name}, welcome to Victoria. Below are your login details:'

    email_data = {
        'app_name': 'Victoria',
        'title': msg_title,
        'body': msg_body,
        'email': email,
        'password': password
    }

    msg = Message(msg_title, sender='support@paydiverse.com', recipients=[email])
    msg.html = render_template('welcome_email_template.html', data=email_data)
    mail.send(msg)


@app.route('/users', methods=['POST'])
@token_required
def add_or_edit_user(current_user):
    data = request.get_json()

    # Extract fields
    user_id = data.get('id')  # Optional for edit
    full_name = data.get('name')
    email = data.get('email')
    password = data.get('password')  # Required only for new users
    role = data.get('role')

    # Validate required fields
    if not full_name or not email or not role or (not user_id and not password):
        return jsonify({'message': 'Missing required fields'}), 400

    if role not in ["super_admin", "agent"]:
        return jsonify({'message': 'Invalid role. Role must be either "super_admin" or "agent".'}), 400

    try:
        cur = mysql.connection.cursor()

        if user_id:
            # Edit existing user
            cur.execute("SELECT COUNT(*) FROM users WHERE id = %s", (user_id,))
            user_exists = cur.fetchone()[0] > 0

            if not user_exists:
                return jsonify({'message': 'User not found.'}), 404

            # Check if the email is used by another user
            cur.execute("SELECT id FROM users WHERE email = %s AND id != %s", (email, user_id))
            email_exists = cur.fetchone()
            if email_exists:
                return jsonify({'message': 'A user with this email already exists.'}), 400

            # Update user details
            query = """
                UPDATE users
                SET name = %s, role = %s, email = %s
                WHERE id = %s
            """
            values = (full_name, role, email, user_id)
            cur.execute(query, values)

        else:
            # Check if the email already exists for new users
            cur.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
            exists = cur.fetchone()[0] > 0

            if exists:
                return jsonify({'message': 'A user with this email already exists.'}), 400

            # Hash the password using bcrypt
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Insert new user
            query = """
                INSERT INTO users (name, password, role, email)
                VALUES (%s, %s, %s, %s)
            """
            values = (full_name, hashed_password.decode('utf-8'), role, email)
            cur.execute(query, values)

            send_welcome_email(email, full_name, password)

        mysql.connection.commit()
        cur.close()

        return jsonify({'message': 'User updated successfully.' if user_id else 'User added successfully.'}), 200

    except Exception as e:
        return jsonify({'message': 'An error occurred.', 'error': str(e)}), 500
    

@app.route('/uploaded-files', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_uploaded_files(current_user):
    date = request.args.get('date')  # Get the date from query parameters

    if not date:
        return jsonify({"error": "Date is required"}), 400
    try:
        cur = mysql.connection.cursor()

        query = '''
            SELECT iso FROM revenue WHERE date = %s GROUP BY iso;
        '''
        cur.execute(query, (date, ))

        results = cur.fetchall()
        cur.close()

        # Format response
        data = [{
            "iso": row[0], 
        } for row in results]

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/users', methods=['GET'])
@token_required
def get_users(current_user):
    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT id, name, role, email
        FROM users
        WHERE id NOT IN (2, 3) AND deleted_at IS NULL;
        '''
        cur.execute(query, )
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'id': row[0],
                'name': row[1],
                'role': row[2],
                'email': row[3]
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/users/<string:id>', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_user_by_mid(current_user, id):
    cur = mysql.connection.cursor()
    
    # Query to fetch merchant by MID
    cur.execute('SELECT name, email, role FROM users WHERE id = %s', (id,))
    
    # Fetch the result
    data = cur.fetchone()
    
    # Check if the merchant exists
    if data:
        column_names = [i[0] for i in cur.description]
        result = dict(zip(column_names, data))
        cur.close()
        return jsonify(result), 200  # Return JSON response with HTTP status 200
    else:
        cur.close()
        return jsonify({'message': 'User not found'}), 200  # Return 404 if MID not found
    

@app.route('/iso', methods=['GET'])
@token_required
def get_iso(current_user):
    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT id, iso, referral_type, is_active
        FROM iso;
        '''
        cur.execute(query, )
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'id': row[0],
                'iso': row[1],
                'referral_type': row[2],
                'is_active': row[3],
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/iso/<string:id>', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_iso_by_id(current_user, id):
    cur = mysql.connection.cursor()
    
    # Query to fetch merchant by MID
    cur.execute('SELECT iso, referral_type, is_active FROM iso WHERE id = %s', (id,))
    
    # Fetch the result
    data = cur.fetchone()
    
    # Check if the merchant exists
    if data:
        column_names = [i[0] for i in cur.description]
        result = dict(zip(column_names, data))
        cur.close()
        return jsonify(result), 200  # Return JSON response with HTTP status 200
    else:
        cur.close()
        return jsonify({'message': 'ISO not found'}), 200  # Return 404 if MID not found


@app.route('/iso', methods=['POST'])
@token_required
def add_or_edit_iso(current_user):
    data = request.get_json()

    # Extract fields
    id = data.get('id')
    iso = data.get('iso')  # Optional for edit
    referral_type = data.get('referral_type')
    is_active = data.get('is_active')

    # Validate required fields
    if not iso or not referral_type:
        return jsonify({'message': 'Missing required fields'}), 400

    if referral_type not in ["MID", "Gateway", "3rd Party"]:
        return jsonify({'message': 'Invalid referral type.'}), 400

    try:
        cur = mysql.connection.cursor()

        if id:
            # Edit existing user
            cur.execute("SELECT COUNT(*) FROM iso WHERE id = %s", (id,))
            iso_exists = cur.fetchone()[0] > 0

            if not iso_exists:
                return jsonify({'message': 'ISO not found.'}), 404

            # Update user details
            query = """
                UPDATE iso
                SET iso = %s, referral_type = %s, is_active = %s, updated_by = %s
                WHERE id = %s
            """
            values = (iso, referral_type, is_active, current_user, id)
            cur.execute(query, values)

        else:
            # Insert new user
            query = """
                INSERT INTO iso (iso, referral_type, is_active, updated_by)
                VALUES (%s, %s, %s, %s)
            """
            values = (iso, referral_type, is_active, current_user)
            cur.execute(query, values)

        mysql.connection.commit()
        cur.close()

        return jsonify({'message': 'ISO updated successfully.' if id else 'ISO added successfully.'}), 200

    except Exception as e:
        return jsonify({'message': 'An error occurred.', 'error': str(e)}), 500


@app.route('/delete-user', methods=['DELETE'])
@token_required
def delete_user(current_user):
    data = request.get_json()
    id = data.get('id')

    if not id:
        return jsonify({'message': 'User ID is required'}), 400

    try:
        cur = mysql.connection.cursor()
        
        # Check if the user exists and is not already deleted
        cur.execute("SELECT COUNT(*) FROM users WHERE id = %s AND deleted_at IS NULL", (id,))
        user_exists = cur.fetchone()[0] > 0
        
        if not user_exists:
            return jsonify({'message': 'User not found or already deleted.'}), 404
        
        # Soft delete user by setting deleted_at and deleted_by
        query = """
            UPDATE users 
            SET deleted_at = NOW(), deleted_by = %s 
            WHERE id = %s
        """
        cur.execute(query, (current_user, id))
        mysql.connection.commit()
        cur.close()

        return jsonify({'message': 'User deleted successfully'}), 200
    
    except Exception as e:
        return jsonify({'message': 'An error occurred.', 'error': str(e)}), 500
    

@app.route('/delete-iso', methods=['DELETE'])
@token_required
def delete_iso(current_user):
    data = request.get_json()
    required_fields = ['date', 'iso']
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        return jsonify({'message': f'Missing fields: {", ".join(missing_fields)}'}), 400

    try:
        date = data['date']
        iso = data['iso']

        cur = mysql.connection.cursor()

        # Delete from revenue table
        delete_query = "DELETE FROM revenue WHERE date = %s AND iso = %s"
        cur.execute(delete_query, (date, iso))

        affected_rows = cur.rowcount

        if affected_rows == 0:
            cur.close()
            return jsonify({'message': 'ISO entry not found'}), 404

        # If deletion was successful, insert into history table
        insert_history_query = """
        INSERT INTO iso_deletion_history (date, iso, deleted_by)
        VALUES (%s, %s, %s)
        """
        cur.execute(insert_history_query, (date, iso, current_user))

        mysql.connection.commit()
        cur.close()

        return jsonify({'message': "ISO's deleted successfully!"}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/adjustments', methods=['GET'])
@token_required
def get_adjustments(current_user):
    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT a.id, i.iso, a.adjustment_price, a.date
        FROM adjustments AS a
        INNER JOIN iso AS i ON i.id =  a.iso_id;
        '''
        cur.execute(query, )
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'id': row[0],
                'iso': row[1],
                'adjustment_price': float(row[2]),
                'date': row[3],
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/adjustments/<string:id>', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_adjustment_by_id(current_user, id):
    cur = mysql.connection.cursor()
    
    # Query to fetch merchant by ID
    cur.execute('SELECT iso_id, adjustment_price, date FROM adjustments WHERE id = %s', (id,))
    
    # Fetch the result
    data = cur.fetchone()
    
    # Check if the merchant exists
    if data:
        column_names = [i[0] for i in cur.description]
        result = dict(zip(column_names, data))
        cur.close()
        return jsonify(result), 200  # Return JSON response with HTTP status 200
    else:
        cur.close()
        return jsonify({'message': 'Record not found'}), 200  # Return 404 if MID not found


@app.route('/adjustments', methods=['POST'])
@token_required
def add_or_edit_adjustments(current_user):
    data = request.get_json()

    # Extract fields
    id = data.get('id')
    iso_id = data.get('iso_id')  # Optional for edit
    date = data.get('date')
    adjustment_price = data.get('adjustment_price')

    # Validate required fields
    if not iso_id or not date or not adjustment_price:
        return jsonify({'message': 'Missing required fields'}), 400

    try:
        cur = mysql.connection.cursor()

        if id:
            # Edit existing user
            cur.execute("SELECT COUNT(*) FROM adjustments WHERE id = %s", (id,))
            adjustment_exists = cur.fetchone()[0] > 0

            if not adjustment_exists:
                return jsonify({'message': 'Record not found.'}), 404

            # Update user details
            query = """
                UPDATE adjustments
                SET iso_id = %s, date = %s, adjustment_price = %s, updated_by = %s
                WHERE id = %s
            """
            values = (iso_id, date, adjustment_price, current_user, id)
            cur.execute(query, values)

        else:
            # Check if an adjustment already exists for the given iso_id and date
            cur.execute("SELECT COUNT(*) FROM adjustments WHERE iso_id = %s AND date = %s", (iso_id, date))
            existing_adjustment = cur.fetchone()[0] > 0

            if existing_adjustment:
                return jsonify({'message': 'An adjustment for this ISO and date already exists.'}), 400
            # Insert new user
            query = """
                INSERT INTO adjustments (iso_id, date, adjustment_price, updated_by)
                VALUES (%s, %s, %s, %s)
            """
            values = (iso_id, date, adjustment_price, current_user)
            cur.execute(query, values)

        mysql.connection.commit()
        cur.close()

        return jsonify({'message': 'Record updated successfully.' if id else 'Record added successfully.'}), 200

    except Exception as e:
        return jsonify({'message': 'An error occurred.', 'error': str(e)}), 500


@app.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({'success': False, 'message': 'Email is required'}), 400

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT name FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

        if not user:
            cur.close()
            return jsonify({'success': False, 'message': 'User not found'}), 404

        # Generate a random 6-digit OTP
        otp = ''.join(random.choices(string.digits, k=6))
        otp_expiry = datetime.datetime.now() + datetime.timedelta(minutes=5)  # OTP valid for 5 mins

        cur.execute(
            "UPDATE users SET otp = %s, otp_expiry = %s WHERE email = %s",
            (otp, otp_expiry, email)
        )
        mysql.connection.commit()
        cur.close()

        msg_title = 'Reset Password OTP Code'
        msg_body = 'Your OTP code for resetting the password is below, do not share it with anyone.'

        email_data = {
            'app_name': 'Victoria',
            'title': msg_title,
            'body': msg_body,
            'otp': otp
        }

        # Send Email
        msg = Message(msg_title, sender='support@paydiverse.com', recipients=[email])
        msg.html = render_template('otp_email_template.html', data=email_data)
        mail.send(msg)

        # Return success with a boolean value
        return jsonify({'success': True, 'message': 'OTP sent successfully'}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    email = data.get('email')
    otp = data.get('otp')

    if not email or not otp:
        return jsonify({'message': 'Email and OTP are required'}), 400

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT otp, otp_expiry FROM users WHERE email = %s", (email,))
        user_data = cur.fetchone()

        if not user_data:
            cur.close()
            return jsonify({'message': 'User not found'}), 404

        stored_otp, otp_expiry = user_data

        # Check if OTP matches and is still valid
        if stored_otp != otp:
            cur.close()
            return jsonify({'message': 'Invalid OTP'}), 400
        
        if datetime.datetime.now() > otp_expiry:
            cur.close()
            return jsonify({'message': 'OTP has expired'}), 400

        # If OTP is correct, clear it after verification
        cur.execute("UPDATE users SET otp = NULL, otp_expiry = NULL WHERE email = %s", (email,))
        mysql.connection.commit()
        cur.close()

        return jsonify({'message': 'OTP verified successfully', 'success': True}), 200

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500
    

@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email')
    new_password = data.get('password')

    if not email or not new_password:
        return jsonify({'message': 'Email and new password are required'}), 400

    try:
        cur = mysql.connection.cursor()

        # Hash the new password before storing it
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        # Update password in the database
        cur.execute('UPDATE users SET password = %s WHERE email = %s', (hashed_password, email))
        mysql.connection.commit()
        cur.close()

        return jsonify({'message': 'Password reset successful', 'success': True}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/revenue-per-month-each-agent', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_revenue_per_month_each_agent(current_user):
    start_date = request.args.get('start_date')  # Get the date from query parameters
    end_date = request.args.get('end_date')  # Get the date from query parameters
    agent_name = request.args.get('agent_name')    # Get the optional iso parameter

    if not start_date and end_date:
        return jsonify({"error": "Start date and end date parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()

        # Query for a specific ISO
        query = '''
            SELECT 
                SUM(
                    CASE 
                        WHEN agent1_name = %s THEN agent1_payout 
                        WHEN agent2_name = %s THEN agent2_payout 
                        ELSE 0 
                    END
                ) AS total_agent_payout, 
                date 
            FROM revenue 
            WHERE (agent1_name = %s OR agent2_name = %s) 
            AND date BETWEEN %s AND %s
            GROUP BY date;

        '''
        cur.execute(query, (agent_name, agent_name, agent_name, agent_name, start_date, end_date))

        results = cur.fetchall()
        cur.close()

        # Format response
        data = [{"date": row[1], "total_revenue": float(row[0])} for row in results]

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/revenue-per-month-each-mid', methods=['GET'])
@token_required  # Apply the token validation decorator
def get_revenue_per_month_each_mid(current_user):
    start_date = request.args.get('start_date')  # Get the date from query parameters
    end_date = request.args.get('end_date')  # Get the date from query parameters
    mid = request.args.get('mid')    # Get the optional iso parameter

    if not start_date and end_date and mid:
        return jsonify({"error": "Start date, end date and MID parameters are required"}), 400

    try:
        cur = mysql.connection.cursor()

        # Query for a specific ISO
        query = '''
            SELECT SUM(paydiverse_residual) AS total_revenue, date 
            FROM revenue 
            WHERE mid = %s AND date between %s and %s
            GROUP BY date
        '''
        cur.execute(query, (mid, start_date, end_date, ))

        results = cur.fetchall()
        cur.close()

        # Format response
        data = [{"date": row[1], "total_revenue": float(row[0])} for row in results]

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/payments', methods=['GET'])
@token_required
def get_payments(current_user):
    try:
        date = request.args.get('date')
    
        if not date:
            return jsonify({"error": "Date is required"}), 400

        cur = mysql.connection.cursor()
        query = '''
        SELECT 
            i.id,
            i.iso, 
            SUM(r.paydiverse_residual) + 
            COALESCE(
                (SELECT SUM(a.adjustment_price) 
                FROM adjustments a 
                WHERE i.id = a.iso_id AND r.date = a.date), 0
            ) AS paydiverse_residual,
            COALESCE(SUM(DISTINCT p.bank_amount), 0) AS bank_amount  -- Fix: Sum only distinct bank_amount values
        FROM revenue r
        LEFT JOIN iso i ON r.iso = i.iso
        LEFT JOIN adjustments a ON i.id = a.iso_id AND r.date = a.date
        LEFT JOIN payments p ON i.id = p.iso_id AND r.date = p.date  -- Join with payments table on iso_id and date
        WHERE r.date = %s
        AND i.iso IS NOT NULL 
        GROUP BY i.id, i.iso  
        ORDER BY 1;
        '''
        cur.execute(query, (date, ))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'id': row[0],
                'iso': row[1],
                'paydiverse_residual': float(row[2]),
                'bank_amount': float(row[3]),
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/payments', methods=['POST'])
@token_required
def add_payments(current_user):
    data = request.get_json()

    # Extract fields
    iso_id = data.get('iso_id')
    date = data.get('date')
    bank_amount = data.get('bank_amount')

    # Validate required fields
    if not iso_id or not date or not bank_amount:
        return jsonify({'message': 'Missing required fields'}), 400

    try:
        cur = mysql.connection.cursor()

        # Check if a record already exists for the given iso_id and date
        cur.execute("SELECT id FROM payments WHERE iso_id = %s AND date = %s", (iso_id, date))
        existing_record = cur.fetchone()

        if existing_record:
            # Update the existing record
            query = """
                UPDATE payments
                SET bank_amount = %s, updated_by = %s
                WHERE iso_id = %s AND date = %s
            """
            values = (bank_amount, current_user, iso_id, date)
            cur.execute(query, values)
            message = "Record updated successfully."
        else:
            # Insert a new record
            query = """
                INSERT INTO payments (iso_id, bank_amount, date, updated_by)
                VALUES (%s, %s, %s, %s)
            """
            values = (iso_id, bank_amount, date, current_user)
            cur.execute(query, values)
            message = "Record added successfully."

        mysql.connection.commit()
        cur.close()

        return jsonify({'message': message}), 200

    except Exception as e:
        return jsonify({'message': 'An error occurred.', 'error': str(e)}), 500


@app.route('/mids-per-iso', methods=['GET'])
@token_required
def get_mids_per_iso(current_user):
    iso = request.args.get('iso')
    
    if not iso:
        return jsonify({"error": "ISO parameter is required"}), 400

    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT mid, iso, dba, corporation, is_active, is_referred, iso_referral_type, approval_date, closed_date 
        FROM merchants 
        WHERE iso = %s
        '''
        cur.execute(query, (iso,))
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'mid': row[0],
                'iso': row[1],
                'dba': row[2],
                'corporation': row[3],
                'is_active': int(row[4]),
                'is_referred': int(row[5]),
                'iso_referral_type': row[6],
                'approval_date': row[7],
                'closed_date': row[8]
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/sic-codes', methods=['GET'])
@token_required
def get_sic_codes(current_user):
    try:
        cur = mysql.connection.cursor()
        query = '''
        SELECT four_digit_sic_codes AS 'sic_code', four_digit_sic_code_descriptions AS 'description' FROM sic_codes 
        '''
        cur.execute(query, )
        results = cur.fetchall()
        cur.close()

        if results:
            data = [{
                'sic_code': row[0],
                'description': row[1],
            } for row in results]
            return jsonify(data), 200
        else:
            return jsonify([]), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()  # Expecting JSON data (email and password)
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password are required'}), 400

    try:
        cur = mysql.connection.cursor()
        # Fetch user from the database by email
        cur.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cur.fetchone()

        if user:
            deleted_at = user[10]  # Assuming deleted_at is in column index 5
            if deleted_at:
                return jsonify({'message': 'User does not exist'}), 404

            stored_password_hash = user[2].encode('utf-8')  # Password hash stored in the database
            user_data = {
                'id': user[0],  # Assuming user ID is in column index 0
                'name': user[1],  # Assuming user's name is in column index 1
                'role': user[3],  # Assuming user role is in column index 3
                'email': user[4]  # Assuming email is in column index 4
            }

            # Check if the password matches using bcrypt.checkpw()
            if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash):
                # Update last_login timestamp
                cur.execute('UPDATE users SET last_login = NOW() WHERE email = %s', (email,))
                mysql.connection.commit()

                # Create JWT token
                token = jwt.encode(user_data, app.config['SECRET_KEY'], algorithm='HS256')

                return jsonify({
                    'message': 'Login successful',
                    'token': token,
                    'user': user_data  # Sending user details as an object
                }), 200
            else:
                return jsonify({'message': 'Invalid email or password'}), 401
        else:
            return jsonify({'message': 'User not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        cur.close()
    

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=5000)