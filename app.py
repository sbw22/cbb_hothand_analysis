import hothand_v2

app = hothand_v2.main()
server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8080)