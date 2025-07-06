import json
import os
from mangum import Mangum
from api.main import app

# Create Mangum handler for FastAPI
handler = Mangum(app)

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    try:
        # Handle the event through Mangum
        response = handler(event, context)
        return response
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'detail': str(e)
            })
        }