from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# 数据库名称列表
DATABASES = ['academic', 'activity_1', 'aircraft', 'allergy_1', 'apartment_rentals', 'architecture', 'assets_maintenance',
             'baseball_1', 'battle_death', 'behavior_monitoring', 'bike_1', 'body_builder', 'book_2', 'browser_web',
             'candidate_poll', 'car_1', 'chinook_1', 'cinema', 'city_record', 'climbing', 'club_1', 'coffee_shop',
             'college_1', 'college_2', 'college_3', 'company_1', 'company_employee', 'company_office', 'concert_singer',
             'county_public_safety', 'course_teach', 'cre_Docs_and_Epenses', 'cre_Doc_Control_Systems', 'cre_Doc_Template_Mgt',
             'cre_Doc_Tracking_DB', 'cre_Drama_Workshop_Groups', 'cre_Theme_park', 'csu_1', 'culture_company', 'customers_and_addresses',
             'customers_and_invoices', 'customers_and_products_contacts', 'customers_campaigns_ecommerce', 'customers_card_transactions',
             'customer_complaints', 'customer_deliveries', 'debate', 'decoration_competition', 'department_management',
             'department_store', 'device', 'document_management', 'dog_kennels', 'dorm_1', 'driving_school', 'election',
             'election_representative', 'employee_hire_evaluation', 'entertainment_awards', 'entrepreneur', 'epinions_1',
             'e_government', 'e_learning', 'farm', 'film_rank', 'flight_1', 'flight_2', 'flight_4', 'flight_company',
             'formula_1', 'game_1', 'game_injury', 'gas_company', 'geo', 'gymnast', 'hospital_1', 'hr_1', 'icfp_1',
             'imdb', 'inn_1', 'insurance_and_eClaims', 'insurance_fnol', 'insurance_policies', 'journal_committee',
             'loan_1', 'local_govt_and_lot', 'local_govt_in_alabama', 'local_govt_mdm', 'machine_repair', 'manufactory_1',
             'manufacturer', 'match_season', 'medicine_enzyme_interaction', 'mountain_photos', 'movie_1', 'museum_visit',
             'musical', 'music_1', 'music_2', 'music_4', 'network_1', 'network_2', 'news_report', 'orchestra', 'party_host',
             'party_people', 'performance_attendance', 'perpetrator', 'pets_1', 'phone_1', 'phone_market', 'pilot_record',
             'poker_player', 'products_for_hire', 'products_gen_characteristics', 'product_catalog', 'program_share',
             'protein_institute', 'race_track', 'railway', 'real_estate_properties', 'restaurants', 'restaurant_1',
             'riding_club', 'roller_coaster', 'sakila_1', 'scholar', 'school_bus', 'school_finance', 'school_player',
             'scientist_1', 'ship_1', 'ship_mission', 'shop_membership', 'singer', 'small_bank_1', 'soccer_1', 'soccer_2',
             'solvency_ii', 'sports_competition', 'station_weather', 'store_1', 'store_product', 'storm_record', 'student_1',
             'student_assessment', 'student_transcripts_tracking', 'swimming', 'theme_gallery', 'tracking_grants_for_research',
             'tracking_orders', 'tracking_share_transactions', 'tracking_software_problems', 'train_station', 'tvshow',
             'twitter_1', 'university_basketball', 'voter_1', 'voter_2', 'wedding', 'wine_1', 'workshop_paper', 'world_1',
             'wrestler', 'wta_1', 'yelp']



def execute_sql(sql, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(sql)
    predicted_res = cursor.fetchall()
    print(f"set(predicted_res): {set(predicted_res)}")
    cursor.close()
    res = ''
    total = len(predicted_res)
    for o in predicted_res:
        res += f"{str(o)}\n\n"
    if res == '':
        res = '未查询到结果'

    if total > 5:
        res += f"total records = {total}\n"
    return res


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取用户选择的数据库和输入的 SQL
        database = request.form.get('database')
        sql = request.form.get('sql')

        db_path = f'../data/spider/database/{database}/{database}.sqlite'

        # 执行 SQL 查询
        result = '查询异常'
        try:
            result = execute_sql(sql, db_path)
        except Exception as e:
            result = str(e)
        return render_template('index.html', databases=DATABASES, result=result, sql=sql, selected_database=database)

    # 渲染初始页面
    return render_template('index.html', databases=DATABASES, sql='', selected_database='')


if __name__ == '__main__':
    app.run(debug=True, port=5002)