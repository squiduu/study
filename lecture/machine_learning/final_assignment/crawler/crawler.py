import requests
import json
import re
from pymongo import MongoClient, UpdateMany


def parse_hotel(region_code, checkin_date, checkout_date):
    print("=" * 20, "Crawler Started", "=" * 20)

    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
    }

    url = "https://www.yanolja.com/api/v1/contents/search"

    parameter = {
        #     'advert':'AREA',
        #     'lat':'37.50681',
        #     'lng':'127.06624',
        #     'rentType':'1',
        #     'stayType':'1'
        "capacityAdult": "2",
        "capacityChild": "0",
        "checkinDate": checkin_date,
        "checkoutDate": checkout_date,
        "hotel": "1",
        "page": "1",
        "region": str(region_code),
        "limit": "50",
        "searchType": "hotel",
        "sort": "133",
    }

    hotel_info = {}

    while True:
        request = requests.get(url, headers=header, params=parameter)
        hotels = json.loads(request.text)

        for hotel in hotels["motels"]["lists"]:
            code = hotel["key"]
            grade = hotel["gradeInfo"]["title"]
            name = hotel["name"]
            address = hotel["addr1"]
            location = hotel["locationDesc"]

            hotel_info[code] = {
                "grade": grade,
                "name": name,
                "address": address,
                "location": location,
            }

        if hotels["paging"]["isLast"]:
            break

        else:
            parameter["page"] = str(int(parameter["page"]) + 1)

    parse_review(hotel_info)

    print("=" * 20, "Crawler Finished", "=" * 20)


def parse_review(hotel_list):
    client = MongoClient(
        "mongodb+srv://NLP:likelion@likelion.hppms.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
    )
    db = client["NLP"]
    collection = db["Nolja"]
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
    }

    for code in list(hotel_list.keys()):
        url = (
            "https://domestic-order-site.yanolja.com/dos-server/review/properties/"
            + code
            + "/reviews"
        )

        parameter = {"size": "50", "sort": "best:desc", "page": "1"}
        page_num = 1

        while True:
            review_list = []
            request = requests.get(url, headers=header, params=parameter)
            reviews = json.loads(request.text)

            for review in reviews["reviews"]:
                id = review["id"]
                date = review["createdAt"][:10]
                room = review["product"]["roomTypeName"]
                user = review["member"]["nickname"]
                content = review["userContent"]["content"]
                total_score = review["userContent"]["totalScore"]
                kindness = review["userContent"]["scores"][0]["score"]
                cleanliness = review["userContent"]["scores"][1]["score"]
                convenience = review["userContent"]["scores"][2]["score"]
                position = review["userContent"]["scores"][3]["score"]

                reivew_data = {
                    "hotelName": hotel_list[code]["name"],
                    "hotelAddr": hotel_list[code]["address"],
                    "hotelLoc": hotel_list[code]["location"],
                    "hotelGrade": hotel_list[code]["grade"],
                    "reviewID": id,
                    "date": date,
                    "room": room,
                    "user": user,
                    "content": content,
                    "totalScore": total_score,
                    "kindness": kindness,
                    "cleanliness": cleanliness,
                    "convenience": convenience,
                    "position": position,
                }
                review_list.append(reivew_data)
            print("=" * 20, page_num, "=" * 20)
            page_num += 1

            if reviews["meta"]["isEnd"]:
                break
            else:
                parameter["page"] = str(int(parameter["page"]) + 1)
            collection.insert_many(review_list)
    client.close()


if __name__ == "__main__":
    parse_hotel(
        region_code="900582", checkin_date="2021-11-01", checkout_date="2021-11-02"
    )
