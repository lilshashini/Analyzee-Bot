import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()
def test_fixed_config():
    """Test only the gpt-4.1 deployment"""
    
    deployment = "gpt-4.1"
    print(f"Testing deployment: {deployment}")
    
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=deployment,  # direct gpt-4.1
            temperature=0,
            max_tokens=800,
        )
        
        response = llm.invoke("Hello, test message")
        print(f"✅ SUCCESS with deployment: {deployment}")
        print(f"Response: {response.content}")
        return deployment
            
    except Exception as e:
        print(f"❌ Failed with {deployment}: {str(e)}")
        if "404" in str(e):
            print("   → This deployment doesn't exist")
        elif "401" in str(e):
            print("   → API key issue")
        elif "Could not resolve host" in str(e):
            print("   → DNS/endpoint issue - check if endpoint is correct")
    
    return None

if __name__ == "__main__":
    working_deployment = test_fixed_config()
    if working_deployment:
        print(f"\n🎯 UPDATE YOUR .ENV FILE WITH:")
        print(f"AZURE_OPENAI_DEPLOYMENT_NAME={working_deployment}")
