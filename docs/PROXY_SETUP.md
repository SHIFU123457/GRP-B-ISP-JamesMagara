# Proxy Setup Guide for Telegram Bot

## Why You Need a Proxy

If you cannot access `https://api.telegram.org` in your browser, Telegram is blocked in your network/region. You need a proxy or VPN to connect.

## Option 1: VPN (Recommended for Development)

1. Install a VPN client (ProtonVPN, NordVPN, etc.)
2. Connect to a server in a region where Telegram is accessible
3. Run your bot - no additional configuration needed

## Option 2: SOCKS5 Proxy

### Using a Proxy Service

1. Sign up for a proxy service or find a reliable proxy
2. Get your proxy credentials (address, port, username, password)
3. Add to your `.env` file:

```bash
# SOCKS5 with authentication
TELEGRAM_PROXY_URL=socks5://username:password@proxy.example.com:1080

# SOCKS5 without authentication
TELEGRAM_PROXY_URL=socks5://proxy.example.com:1080
```

### Free Public Proxies (Not Recommended for Production)

Search for "free socks5 proxy list" and test proxies. Example:

```bash
TELEGRAM_PROXY_URL=socks5://123.45.67.89:1080
```

**Warning:** Free proxies are often unreliable, slow, and may log your traffic.

## Option 3: HTTP Proxy

If you have an HTTP proxy:

```bash
# HTTP with authentication
TELEGRAM_PROXY_URL=http://username:password@proxy.example.com:8080

# HTTP without authentication
TELEGRAM_PROXY_URL=http://proxy.example.com:8080
```

## Option 4: SSH Tunnel (Advanced)

If you have SSH access to a server in another region:

### Step 1: Create SSH Tunnel

```bash
# Create a SOCKS5 proxy on local port 1080
ssh -D 1080 -C -N user@your-server.com
```

Keep this terminal open while running your bot.

### Step 2: Configure Bot

Add to `.env`:

```bash
TELEGRAM_PROXY_URL=socks5://127.0.0.1:1080
```

### Step 3: Run Your Bot

```bash
python src/main.py
```

## Option 5: MTProto Proxy (Telegram-Specific)

MTProto proxies are specifically designed for Telegram:

1. Find or set up an MTProto proxy
2. Unfortunately, python-telegram-bot doesn't support MTProto directly
3. You'll need to use SOCKS5 or HTTP proxy instead

## Testing Your Proxy

### Test Proxy Connection

```python
# test_proxy.py
import asyncio
import httpx

async def test_proxy():
    proxy_url = "socks5://your-proxy:port"  # Change this

    async with httpx.AsyncClient(proxy=proxy_url, timeout=30.0) as client:
        try:
            response = await client.get("https://api.telegram.org/bot")
            print(f"✅ Proxy works! Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Proxy failed: {e}")

asyncio.run(test_proxy())
```

## Troubleshooting

### "Connection refused" or "Connection timeout"
- Check proxy address and port
- Verify proxy is running and accessible
- Check firewall settings

### "Authentication failed"
- Verify username and password
- Check proxy URL format: `socks5://user:pass@host:port`

### "Proxy protocol not supported"
- Ensure you're using `socks5://` or `http://` prefix
- Check if your proxy supports the protocol you're using

### Still not working?
- Try a different proxy
- Use a VPN instead
- Check if you need to install additional dependencies:

```bash
pip install httpx[socks]
```

## Recommended Proxy Services (Paid)

- **Bright Data** - Reliable, many locations
- **Smartproxy** - Good for bots
- **Oxylabs** - Enterprise-grade
- **ProxyRack** - Budget-friendly

## Security Considerations

⚠️ **Important:**
- Never use untrusted proxies with sensitive data
- Free proxies may log your traffic
- Use encrypted connections (SOCKS5 over TLS when possible)
- Rotate proxies for production use
- Monitor proxy health and failover

## Example `.env` Configuration

```bash
# Required
TELEGRAM_BOT_TOKEN=your_bot_token_here
DATABASE_URL=your_database_url_here

# Proxy Configuration (choose one)
TELEGRAM_PROXY_URL=socks5://username:password@proxy.example.com:1080

# Network Timeouts (already configured with good defaults)
TELEGRAM_CONNECT_TIMEOUT=60.0
TELEGRAM_READ_TIMEOUT=60.0
TELEGRAM_WRITE_TIMEOUT=60.0
TELEGRAM_POOL_TIMEOUT=10.0
```

## Need Help?

If you're still having issues:
1. Verify your proxy works with other applications
2. Test with the test script above
3. Check application logs for specific error messages
4. Consider using a VPN as a simpler alternative
