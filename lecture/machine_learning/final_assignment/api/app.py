from os import access
import bcrypt
from dns import message
from flask import Flask, request, Response, jsonify, redirect, url_for, make_response
import json
from flask_jwt_extended.utils import get_jwt_identity, set_access_cookies, set_refresh_cookies
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt, generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, create_refresh_token
from operator import itemgetter
from flask_cors import CORS, cross_origin
from werkzeug.datastructures import Headers


app = Flask(__name__)

app.config['MONGO_URI'] = 'mongodb+srv://NLP:likelion@likelion.hppms.mongodb.net/NLP'
app.config["JWT_SECRET_KEY"] = "YourStay"
app.config["CORS_HEADERS"] = ['Content-Type, auth']
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
# app.config['JWT_ACCESS_COOKIE_PATH'] = '/recommnded'
# app.config['JWT_COOKIE_SECURE'] = False 
# app.config['JWT_COOKIE_CSRF_PROTECT'] = False
# app.config['MONGO_DBNAME'] = 'NLP'

mongo = PyMongo(app)
bcrpyt = Bcrypt(app)
jwt = JWTManager(app)
cors = CORS(app, resources={r'/':{'origins':'http://localhost:5000'}}, support_credentials=True)

db = mongo.db

@app.route('/api/v1/register', methods=['POST'])
@cross_origin(origin='localhost', support_credentials=True)
def register():
    user_id = request.form['userid']
    user_name = request.form['username']
    password = request.form['password']
    clean = request.form['clean']
    conv = request.form['conv']
    kind = request.form['kind']
    position = request.form['position']

    if clean==0 and conv==0 and kind==0 and position==0:
        return jsonify(message="Please Enter Score")


    if db.user.find_one({'user_id':user_id}):
        return jsonify(message='ID Already Exist')

    else:
        hased = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        total_sum = int(clean) + int(conv) + int(kind) + int(position)
        user_data = {'user_id':user_id, 'password':hased, 'user_name':user_name, 'cleanliness':(int(clean)/total_sum)*100, 'convenience':(int(conv)/total_sum)*100, 'kindness':(int(kind)/total_sum)*100, 'position':(int(position)/total_sum)*100}
        db.user.insert_one(user_data)

        return jsonify(message='User Registered')


@app.route('/api/v1/login', methods=['POST', 'OPTIONS'])
@cross_origin(origin='localhost', support_credentials=True)
def login():

    if request.is_json:
        user_id = request.json['userid']
        password = request.json['password']
    else:
        user_id = request.form['userid']
        password = request.form['password']

    
    user_info = db.user.find_one({'user_id':user_id})

    if not user_info:
        return jsonify(message='No User')

    else:
        id_val = user_info['user_id']
        pwcheck = user_info['password']
        
        if bcrypt.checkpw(password.encode('utf-8'), pwcheck):
            access_token = create_access_token(identity=id_val)
            refresh_token = create_refresh_token(identity=id_val)

            resp = jsonify(message="Logged In")
            set_access_cookies(resp, access_token)
            # set_refresh_cookies(resp, refresh_token)

            return resp

        else:
            return jsonify(message="No verified email or password")



@app.route('/api/v1/hotels', methods=['GET'])
def get_hotel():
    region = request.args.get("region", "all", str)

    hotels = db.hotel
    output = []

    if region == 'all':

        for h in list(hotels.find()):
            del h['_id']
            output.append(h)
            # output.append({'name':h['hotelName'], 'cleanliness':h['cleanliness'], 'convenience':h['convenience'], 'kindness': h['kindness'], 'position':h['position'], 'score':h['totalScore'], 'id':h['hotel_id'], 'image':h['url']})

        return jsonify({'result':output})

    else:
        for h in list(hotels.find({'$text':{'$search':region}})):
            del h['_id']
            output.append(h)
            # output.append({'name':h['hotelName'], 'cleanliness':h['cleanliness'], 'convenience':h['convenience'], 'kindness': h['kindness'], 'position':h['position'], 'score':h['totalScore'], 'id':h['hotel_id'], 'image':h['url']})

        return jsonify({'result':output})


@app.route('/api/v1/hotels/recommended', methods=['GET'])
@jwt_required()
def recommend_hotel():
    hotels = db.hotel
    users = db.user
    
    current_user = get_jwt_identity()
    user_info = users.find_one({'user_id':current_user})
    user_scores = {'cleanliness':user_info['cleanliness'], 'position':user_info['position'], 'convenience': user_info['convenience'], 'kindness':user_info['kindness']}

    hotel_infos = list(hotels.find())

    weighted_scores = []

    for h in hotel_infos:
        weighted = h
        del weighted['_id']
        # weighted['hotelName'] = h['hotelName']
        weighted['score'] = (user_scores['cleanliness']*h['cleanliness'] + user_scores['position']*h['position'] + user_scores['convenience']*h['convenience'] + user_scores['kindness']*h['kindness']) / 4
        weighted_scores.append(weighted)

    
    recommended = sorted(weighted_scores, key = itemgetter('score'), reverse=True)[0:5]

    return jsonify({'result':recommended})
    

@app.route('/api/v1/hotels/reviews/<int:hotelid>', methods=['GET'])
def get_review(hotelid):
    reviews = db.tokenized_review
    hotel = db.hotel
    keyword = db.keyword

    hotel_doc = hotel.find_one({'hotel_id':hotelid})
    del hotel_doc['_id']

    keyword_doc = list(keyword.find({'hotel_id':hotelid}))

    review_doc = list(reviews.find({'hotel_id':hotelid}))

    keyword_output = []

    for k in keyword_doc:
        keyword_info = {'rep':k['rep']}

        k_list = k['keywords']

        review_content = []
        review_ids = []

        pos_count = 0
        neg_count = 0

        for w in k_list:
            # for r in reviews.find({'hotel_id':hotelid,'tokens':w}):
            for r in review_doc:
                if w in r['tokens']:
                    if r['label'] == 0:
                        neg_count += 1
                    elif r['label'] == 1:
                        pos_count += 1

                    if r['reviewID'] not in review_ids:
                        keyword_reviews = {'id':r['reviewID'], 'content':r['content']}
                        review_content.append(keyword_reviews)
                        review_ids.append(r['reviewID'])
                
        keyword_info['pos'] = pos_count
        keyword_info['neg'] = neg_count
        keyword_info['reviews'] = review_content


        keyword_output.append(keyword_info)

    keyword_output_neg = sorted(keyword_output, key = itemgetter('neg'), reverse=True)[0:3]
    keyword_output_pos = sorted(keyword_output, key = itemgetter('pos'), reverse=True)[0:3]

    return jsonify({'hotel':hotel_doc, 'pos_review':keyword_output_pos, 'neg_review':keyword_output_neg})


if __name__ == "__main__":
    app.run(host='0.0.0.0')
