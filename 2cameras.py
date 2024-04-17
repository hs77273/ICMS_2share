import cv2


def main():
    cap1 = cv2.VideoCapture(2)
    cap2 = cv2.VideoCapture(1)

    if not (cap1.isOpened() and cap2.isOpened()):
        print("Error: Cannot open cameras.")
        return

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            print("Error: Cannot read frame. Retrying...")
            continue

        frame1 = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
        frame2 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)
        
        horizontal_stack = cv2.hconcat([frame1, frame2])

        cv2.imshow("Triple Camera 1 View", horizontal_stack)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
