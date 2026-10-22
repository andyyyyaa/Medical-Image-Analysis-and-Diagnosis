# Production deployment guide

## Diagnosing problems

### Causes of flash message and login session issues

1. **Session configuration**: Incorrect session configuration in production.
2. **Secret key**: The environment variable is not set correctly.
3. **Cookie configuration**: HTTPS/HTTP settings do not match.

## Solutions

### 1. Use the new startup script

```bash
# Use the production startup script
python3 start_server.py
```

### 2. Set environment variables

```bash
# Set the secret key (important!)
export SECRET_KEY="your-very-secure-secret-key-here"

# Set the host and port
export HOST="0.0.0.0"
export PORT="5000"
```

### 3. Check database permissions

Ensure the database file on the server has the appropriate read and write permissions:

```bash
chmod 644 user_db.sqlite3
chmod 755 static/
```

### 4. Configure the firewall

Ensure the server firewall allows port 5000:

```bash
# Ubuntu/Debian
sudo ufw allow 5000

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload
```

## Deployment steps

### 1. Upload the code to the server

```bash
# Upload all files to the server
scp -r ./* user@your-server:/path/to/app/
```

### 2. Install dependencies

```bash
# Install dependencies on the server
pip3 install -r requirements.txt
```

### 3. Set environment variables

```bash
# Edit ~/.bashrc or ~/.profile
echo 'export SECRET_KEY="your-secret-key"' >> ~/.bashrc
echo 'export HOST="0.0.0.0"' >> ~/.bashrc
echo 'export PORT="5000"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Start the application

```bash
# Use the production startup script
python3 start_server.py
```

### 5. Run in the background (recommended)

```bash
# Run in the background with nohup
nohup python3 start_server.py > app.log 2>&1 &

# Alternatively, use screen
screen -S healthcare
python3 start_server.py
# Press Ctrl+A, D to detach from screen
```

## Troubleshooting

### 1. Check logs

```bash
# View application logs
tail -f app.log

# View system logs
sudo journalctl -u your-service -f
```

### 2. Check processes

```bash
# View Python processes
ps aux | grep python

# Check which process is using the port
netstat -tlnp | grep 5000
```

### 3. Test connectivity

```bash
# Local test
curl http://localhost:5000

# Remote test
curl http://your-server-ip:5000
```

## Security recommendations

1. **Use HTTPS**: HTTPS is recommended in production.
2. **Set a strong secret key**: Use a randomly generated, strong key.
3. **Restrict access**: Use a firewall to limit allowed sources.
4. **Back up regularly**: Back up the database and code regularly.
5. **Monitor logs**: Monitor application and system logs.

## Performance optimization

1. **Use Gunicorn**: Gunicorn is recommended for production.
2. **Use an Nginx reverse proxy**: Use Nginx as a frontend proxy.
3. **Optimize the database**: Consider a production database such as PostgreSQL.
4. **Add caching**: Add Redis caching support.
