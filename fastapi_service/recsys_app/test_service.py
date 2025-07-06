### test_service.py
import requests
import logging

logging.basicConfig(
    filename="test_service.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_URL = "http://localhost:8000/recommendations"

TEST_USERS = [992329,
 125625,
 800456,
 428642,
 712443,
 492414,
 948586,
 85734,
 820159,
 1057088,
 1185234]

def run_test_case(user_id: int):
    url = f"{BASE_URL}/{user_id}"
    try:
        response = requests.get(url)
        result = {}
        try:
            result = response.json()
        except Exception as je:
            result = {"raw_response": response.text, "json_error": str(je)}

        logging.info(f"Test case: user_id={user_id}")
        logging.info(f"Status code: {response.status_code}")
        logging.info(f"Response: {result}")

        assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

        if "error" in result:
            logging.warning(f"⚠️ API returned an error: {result['error']}")
            return

        assert "user_id" in result, "Missing 'user_id' field in response"
        assert "recommendations" in result, "Missing 'recommendations' field in response"
        assert isinstance(result["recommendations"], list), "'recommendations' should be a list"
        assert len(result["recommendations"]) <= 10, "Too many recommendations returned"

        logging.info("✅ Test passed.\n")
    except Exception as e:
        logging.error(f"❌ Test failed: {e}\n")

if __name__ == "__main__":
    print("Running tests... Check test_service.log for results.")
    for user_id in TEST_USERS:
        run_test_case(user_id)