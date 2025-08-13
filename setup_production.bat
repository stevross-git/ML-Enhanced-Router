@echo off
REM Production Setup Script for MLEnhancedRouter (Windows)
REM This script sets up the database with Docker and local virtual environment for development

echo MLEnhancedRouter Production Setup
echo ==================================

REM Check prerequisites
echo Checking prerequisites...

where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed. Please install Docker Desktop for Windows.
    echo Visit: https://docs.docker.com/desktop/windows/install/
    exit /b 1
)

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed. Please install Python 3.11 or later.
    exit /b 1
)

echo All prerequisites met!

REM Step 1: Create .env file if it doesn't exist
if not exist .env (
    echo.
    echo Creating .env file from template...
    copy .env.example .env
    echo IMPORTANT: Edit .env file and set secure passwords before proceeding!
    pause
) else (
    echo .env file already exists
)

REM Step 2: Set up Python virtual environment
echo.
echo Setting up Python virtual environment...

if not exist venv (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing Python dependencies...
pip install -r requirements.txt

echo Python dependencies installed

REM Step 3: Start Docker services (Database and Redis only)
echo.
echo Starting Docker services (PostgreSQL and Redis)...

REM Create a docker-compose override for database-only setup
(
echo # Override to run only database and cache services
echo # The application will run locally in virtual environment
echo.
echo services:
echo   # Disable the app service - we'll run it locally
echo   app:
echo     deploy:
echo       replicas: 0
echo     command: echo "App service disabled - running locally"
echo.    
echo   # pgAdmin is optional, comment out if not needed
echo   pgadmin:
echo     profiles:
echo       - with-pgadmin
) > docker-compose.override.yml

REM Start only database and redis services
docker-compose up -d db redis

echo Waiting for database to be ready...
timeout /t 10 /nobreak >nul

echo Database services are running

REM Step 4: Initialize database
echo.
echo Initializing database...

REM Set environment variables for local development
set DATABASE_URL=postgresql://ml_router_user:ml_router_password@localhost:5432/ml_router_db
set SQLALCHEMY_DATABASE_URI=%DATABASE_URL%
set REDIS_URL=redis://localhost:6379/0
set FLASK_ENV=development

REM Initialize the database
python init_db.py

echo Database initialized

REM Step 5: Create run script for local development
(
echo @echo off
echo REM Script to run the application locally with virtual environment
echo.
echo REM Load environment variables from .env file
echo if exist .env (
echo     for /f "tokens=1,2 delims==" %%%%a in ^(.env^) do (
echo         if not "%%%%a"=="" if not "%%%%a:~0,1%"=="#" set %%%%a=%%%%b
echo     ^)
echo ^)
echo.
echo REM Activate virtual environment
echo call venv\Scripts\activate.bat
echo.
echo REM Set database connection to Docker containers
echo set DATABASE_URL=postgresql://%%POSTGRES_USER%%:%%POSTGRES_PASSWORD%%@localhost:5432/%%POSTGRES_DB%%
echo set SQLALCHEMY_DATABASE_URI=%%DATABASE_URL%%
echo set REDIS_URL=redis://localhost:6379/0
echo if not defined FLASK_ENV set FLASK_ENV=development
echo.
echo REM Run the application
echo echo Starting MLEnhancedRouter...
echo echo Application: http://localhost:5000
echo echo Database: PostgreSQL ^(Docker^) on localhost:5432
echo echo Cache: Redis ^(Docker^) on localhost:6379
echo echo.
echo python main.py
) > run_local.bat

REM Step 6: Create management scripts
(
echo @echo off
echo REM Service management script
echo.
echo if "%%1"=="start" goto start
echo if "%%1"=="stop" goto stop
echo if "%%1"=="restart" goto restart
echo if "%%1"=="status" goto status
echo if "%%1"=="logs" goto logs
echo if "%%1"=="pgadmin" goto pgadmin
echo.
echo echo Usage: manage_services.bat {start^|stop^|restart^|status^|logs^|pgadmin}
echo exit /b 1
echo.
echo :start
echo echo Starting database services...
echo docker-compose up -d db redis
echo echo Services started
echo echo Run run_local.bat to start the application
echo exit /b 0
echo.
echo :stop
echo echo Stopping database services...
echo docker-compose stop db redis
echo echo Services stopped
echo exit /b 0
echo.
echo :restart
echo echo Restarting database services...
echo docker-compose restart db redis
echo echo Services restarted
echo exit /b 0
echo.
echo :status
echo echo Service status:
echo docker-compose ps db redis
echo exit /b 0
echo.
echo :logs
echo docker-compose logs -f db redis
echo exit /b 0
echo.
echo :pgadmin
echo echo Starting pgAdmin...
echo docker-compose --profile with-pgadmin up -d pgadmin
echo echo pgAdmin available at http://localhost:8080
echo exit /b 0
) > manage_services.bat

REM Step 7: Display summary
echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Service Status:
docker-compose ps

echo.
echo Next Steps:
echo 1. Database is running in Docker on localhost:5432
echo 2. Redis is running in Docker on localhost:6379
echo 3. To start the application locally:
echo    run_local.bat
echo.
echo Management Commands:
echo    manage_services.bat start    - Start database services
echo    manage_services.bat stop     - Stop database services
echo    manage_services.bat status   - Check service status
echo    manage_services.bat logs     - View service logs
echo    manage_services.bat pgadmin  - Start pgAdmin (optional)
echo.
echo Configuration:
echo    - Edit .env file for API keys and settings
echo    - Database: PostgreSQL (Docker)
echo    - Cache: Redis (Docker)
echo    - Application: Python virtual environment (local)
echo.
echo Access Points:
echo    - Application: http://localhost:5000
echo    - pgAdmin: http://localhost:8080 (if enabled)
echo.
echo For production deployment:
echo    1. Set strong passwords in .env file
echo    2. Configure SSL/TLS certificates
echo    3. Set FLASK_ENV=production
echo    4. Use 'docker-compose up -d' to run full stack in Docker

pause