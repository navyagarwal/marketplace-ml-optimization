"""
Simple script to test the API endpoints
"""
import requests
import json
from typing import Dict

API_BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)

    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_freight_prediction():
    """Test freight prediction endpoint"""
    print("\n" + "=" * 60)
    print("Testing Freight Prediction")
    print("=" * 60)

    payload = {
        "distance_km": 250.5,
        "product_weight_g": 500,
        "product_volume_cm3": 1000,
        "product_category_name": "health_beauty",
        "price": 50.0,
        "order_month": 6,
        "is_weekend": 0
    }

    print(f"Request: {json.dumps(payload, indent=2)}")

    response = requests.post(
        f"{API_BASE_URL}/predict/freight",
        json=payload
    )

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Predicted Freight Cost: {result['prediction_formatted']}")
        return True
    else:
        print(f"\n❌ Error: {response.text}")
        return False


def test_delivery_prediction():
    """Test delivery prediction endpoint"""
    print("\n" + "=" * 60)
    print("Testing Delivery Prediction")
    print("=" * 60)

    payload = {
        "distance_km": 250.5,
        "product_weight_g": 500,
        "product_volume_cm3": 1000,
        "product_category_name": "health_beauty",
        "price": 50.0,
        "order_day_of_week": 2,
        "order_hour": 14,
        "order_month": 6,
        "is_weekend": 0
    }

    print(f"Request: {json.dumps(payload, indent=2)}")

    response = requests.post(
        f"{API_BASE_URL}/predict/delivery",
        json=payload
    )

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Predicted Delivery Time: {result['prediction_formatted']}")
        return True
    else:
        print(f"\n❌ Error: {response.text}")
        return False


def test_various_scenarios():
    """Test with different scenarios"""
    print("\n" + "=" * 60)
    print("Testing Various Scenarios")
    print("=" * 60)

    scenarios = [
        {
            "name": "Short distance, light product",
            "freight": {
                "distance_km": 50,
                "product_weight_g": 100,
                "product_volume_cm3": 200,
                "product_category_name": "toys",
                "price": 20.0,
                "order_month": 3,
                "is_weekend": 0
            },
            "delivery": {
                "distance_km": 50,
                "product_weight_g": 100,
                "product_volume_cm3": 200,
                "product_category_name": "toys",
                "price": 20.0,
                "order_day_of_week": 1,
                "order_hour": 10,
                "order_month": 3,
                "is_weekend": 0
            }
        },
        {
            "name": "Long distance, heavy product",
            "freight": {
                "distance_km": 1000,
                "product_weight_g": 5000,
                "product_volume_cm3": 10000,
                "product_category_name": "furniture_decor",
                "price": 200.0,
                "order_month": 12,
                "is_weekend": 1
            },
            "delivery": {
                "distance_km": 1000,
                "product_weight_g": 5000,
                "product_volume_cm3": 10000,
                "product_category_name": "furniture_decor",
                "price": 200.0,
                "order_day_of_week": 6,
                "order_hour": 20,
                "order_month": 12,
                "is_weekend": 1
            }
        }
    ]

    results = []

    for scenario in scenarios:
        print(f"\n{'─' * 60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'─' * 60}")

        # Test freight
        freight_response = requests.post(
            f"{API_BASE_URL}/predict/freight",
            json=scenario['freight']
        )

        # Test delivery
        delivery_response = requests.post(
            f"{API_BASE_URL}/predict/delivery",
            json=scenario['delivery']
        )

        if freight_response.status_code == 200 and delivery_response.status_code == 200:
            freight_result = freight_response.json()
            delivery_result = delivery_response.json()

            print(f"  Freight Cost: {freight_result['prediction_formatted']}")
            print(f"  Delivery Time: {delivery_result['prediction_formatted']}")

            results.append({
                'scenario': scenario['name'],
                'freight': freight_result['prediction'],
                'delivery': delivery_result['prediction'],
                'success': True
            })
        else:
            print(f"  ❌ Error in prediction")
            results.append({
                'scenario': scenario['name'],
                'success': False
            })

    return results


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ML API TEST SUITE")
    print("=" * 60)
    print(f"Testing API at: {API_BASE_URL}")
    print("=" * 60)

    tests_passed = 0
    total_tests = 3

    try:
        # Test 1: Health check
        if test_health():
            tests_passed += 1
        else:
            print("\n❌ Health check failed - is the API running?")
            print(f"   Start the API with: cd src && python api.py")
            return

        # Test 2: Freight prediction
        if test_freight_prediction():
            tests_passed += 1

        # Test 3: Delivery prediction
        if test_delivery_prediction():
            tests_passed += 1

        # Additional: Various scenarios
        print("\n" + "=" * 60)
        test_various_scenarios()

    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Could not connect to API")
        print(f"   Make sure the API is running at {API_BASE_URL}")
        print(f"   Start with: cd src && python api.py")
        return

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("\n✅ All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - tests_passed} test(s) failed")

    print("\n" + "=" * 60)
    print("To view interactive API docs, visit:")
    print(f"  Swagger UI: {API_BASE_URL}/docs")
    print(f"  ReDoc:      {API_BASE_URL}/redoc")
    print("=" * 60)


if __name__ == "__main__":
    main()
