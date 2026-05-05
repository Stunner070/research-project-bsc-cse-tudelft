#create a demo file outputting the current date and time
import datetime
def main():
    now = datetime.datetime.now()
    with open("demo.txt", "w") as f:
        f.write("Current date and time: " + str(now))
if __name__ == "__main__":    main()