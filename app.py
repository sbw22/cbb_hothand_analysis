import hothand_v2
import os

print("PID:", os.getpid())

if __name__ == "__main__":
    app = hothand_v2.main()
    server = app.server

    app.run(debug=False, port=8080)