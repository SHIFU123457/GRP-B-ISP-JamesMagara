"""
Test script to verify proxy connection to Telegram API
Run this before starting your bot to ensure proxy is working
"""
import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

async def test_proxy():
    """Test if proxy can connect to Telegram API"""

    proxy_url = os.getenv("TELEGRAM_PROXY_URL")
    connect_timeout = float(os.getenv("TELEGRAM_CONNECT_TIMEOUT", "60.0"))

    print("=" * 60)
    print("🔍 Testing Telegram API Connection")
    print("=" * 60)

    if proxy_url:
        print(f"📡 Using proxy: {proxy_url}")
    else:
        print("⚠️  No proxy configured (direct connection)")

    print(f"⏱️  Timeout: {connect_timeout}s")
    print()

    # Test URL
    test_url = "https://api.telegram.org/bot"

    try:
        print(f"🌐 Attempting to connect to: {test_url}")

        # Configure client
        client_kwargs = {
            'timeout': httpx.Timeout(connect_timeout),
        }

        if proxy_url:
            client_kwargs['proxy'] = proxy_url

        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(test_url)

            print()
            print("=" * 60)
            print("✅ SUCCESS! Connection established")
            print("=" * 60)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            print()
            print("🎉 Your proxy/connection is working!")
            print("You can now start your bot with: python src/main.py")
            return True

    except httpx.ConnectError as e:
        print()
        print("=" * 60)
        print("❌ CONNECTION FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("💡 Solutions:")
        if not proxy_url:
            print("  1. You need a proxy or VPN to access Telegram")
            print("  2. Add TELEGRAM_PROXY_URL to your .env file")
            print("  3. Example: TELEGRAM_PROXY_URL=socks5://proxy:1080")
        else:
            print("  1. Verify proxy address and port are correct")
            print("  2. Check if proxy requires authentication")
            print("  3. Try a different proxy")
            print("  4. Use a VPN instead")
        print()
        print("📖 See docs/PROXY_SETUP.md for detailed instructions")
        return False

    except httpx.ProxyError as e:
        print()
        print("=" * 60)
        print("❌ PROXY ERROR")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("💡 Solutions:")
        print("  1. Check TELEGRAM_PROXY_URL format")
        print("  2. Should be: socks5://host:port or http://host:port")
        print("  3. With auth: socks5://user:pass@host:port")
        print("  4. Install SOCKS support: pip install httpx[socks]")
        return False

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ UNEXPECTED ERROR")
        print("=" * 60)
        print(f"Error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print()
    success = asyncio.run(test_proxy())
    print()

    if not success:
        print("⚠️  Fix the connection issues above before running your bot")
        exit(1)
    else:
        exit(0)
