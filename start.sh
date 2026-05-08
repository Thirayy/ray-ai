#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "venv" ]]; then
  echo "❌ venv belum ada. Buat dulu: python3 -m venv venv"
  exit 1
fi

source venv/bin/activate

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${API_PORT:=8000}"
: "${MYSQL_HOST:=127.0.0.1}"
: "${MYSQL_PORT:=3306}"
: "${MYSQL_USER:=rayapp}"
: "${MYSQL_PASSWORD:=adminraya}"
: "${MYSQL_DATABASE:=ray_ai}"

ensure_service_running() {
  local service_name="$1"
  local friendly_name="$2"

  if ! command -v systemctl >/dev/null 2>&1; then
    echo "⚠️  systemctl tidak tersedia. Lewati auto-start $friendly_name."
    return 0
  fi

  if systemctl is-active --quiet "$service_name"; then
    echo "✅ $friendly_name aktif"
    return 0
  fi

  echo "ℹ️  $friendly_name belum aktif, mencoba start..."

  if sudo -n systemctl start "$service_name" >/dev/null 2>&1; then
    :
  else
    sudo systemctl start "$service_name"
  fi

  if systemctl is-active --quiet "$service_name"; then
    echo "✅ $friendly_name berhasil dijalankan"
    return 0
  fi

  echo "❌ Gagal start $friendly_name ($service_name)."
  return 1
}

if ! python -c "import pymysql" >/dev/null 2>&1; then
  echo "❌ Driver MySQL belum ada di venv."
  echo "   Jalankan: ./venv/bin/python -m pip install PyMySQL"
  exit 1
fi

# Start DB service (mysql atau mariadb)
if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-unit-files | grep -qE '^mysql\.service'; then
    ensure_service_running "mysql" "MySQL"
  elif systemctl list-unit-files | grep -qE '^mariadb\.service'; then
    ensure_service_running "mariadb" "MariaDB"
  else
    echo "⚠️  Service MySQL/MariaDB tidak terdeteksi via systemctl."
    echo "   Pastikan database server sudah jalan manual."
  fi

  # Start Apache untuk phpMyAdmin
  if systemctl list-unit-files | grep -qE '^apache2\.service'; then
    ensure_service_running "apache2" "Apache2 (phpMyAdmin)"
  else
    echo "⚠️  Service apache2 tidak terdeteksi. phpMyAdmin mungkin tidak bisa diakses."
  fi
fi

if ! mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" -e "USE \`$MYSQL_DATABASE\`; SELECT 1;" >/dev/null 2>&1; then
  echo "❌ Gagal konek MySQL: $MYSQL_USER@$MYSQL_HOST:$MYSQL_PORT / DB=$MYSQL_DATABASE"
  echo "   Cek .env (MYSQL_HOST/PORT/USER/PASSWORD/DATABASE) dan pastikan MariaDB/MySQL sudah running."
  exit 1
fi

IP="$(hostname -I | awk '{print $1}')"
echo "🚀 Ray AI starting..."
echo "✅ MySQL connected: $MYSQL_USER@$MYSQL_HOST:$MYSQL_PORT/$MYSQL_DATABASE"
echo "🌐 Local   : http://localhost:${API_PORT}/static/index.html"
echo "🌐 Network : http://${IP}:${API_PORT}/static/index.html"
echo "🗄️ DB UI Local   : http://localhost/phpmyadmin"
echo "🗄️ DB UI Network : http://${IP}/phpmyadmin"
echo "   Login DB UI: user=$MYSQL_USER | db=$MYSQL_DATABASE"
echo ""

python ray_ai.py --api
