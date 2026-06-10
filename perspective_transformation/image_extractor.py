import argparse
import cv2
import numpy as np

selected_points = []
img_original = None
img_displayed = None

def mouse_callback(event, x, y, flags, param):
    global selected_points, img_displayed

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 4:
            selected_points.append((x, y))
            cv2.circle(img_displayed, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img_displayed, f"{len(selected_points)}", (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("Select Points", img_displayed)

            if len(selected_points) == 4:
                perform_perspective_transform(param["width"], param["height"])

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def perform_perspective_transform(target_w, target_h):
    global selected_points, img_original

    src_pts = order_points(selected_points)

    dst_pts = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_original, M, (target_w, target_h))

    cv2.imshow("Warped Image", warped)
    print("Perspective transformation completed. Close the windows to exit.")

def main():
    global img_original, img_displayed

    parser = argparse.ArgumentParser(description="Perspective Transformation von Bildern")
    parser.add_argument("-i", "--input", required=True, help="Pfad zum Eingabebild")
    parser.add_argument("-o", "--output", required=True, help="Pfad zum Ausgabeordner")
    parser.add_argument("-r", "--resolution", required=True, help="Zielauflösung (Breite Höhe)")
    args = parser.parse_args()

    try:
        width, height = map(int, args.resolution.lower().split("x"))
    except ValueError:
        print("Ungültiges Auflösungsformat. Bitte verwenden Sie 'Breite Höhe' (z.B. 800x600).")
        return
    
    img_original = cv2.imread(args.input)
    if img_original is None:
        print("Fehler beim Laden des Bildes. Bitte überprüfen Sie den Pfad.")
        return
    
    img_displayed = img_original.copy()

    cv2.namedWindow("Select Points")
    target_res = {"width": width, "height": height}
    cv2.setMouseCallback("Select Points", mouse_callback, param=target_res)

    print("Bitte wählen Sie 4 Punkte im Bild aus, um die Perspektivtransformation durchzuführen.")
    print("ESC = Zurücksetzen | S = Ergebnis speichern | Q = Beenden")

    while True:
        cv2.imshow("Select Points", img_displayed)
        key = cv2.waitKey(1) & 0xFF

        if key == 27: 
            selected_points.clear()
            img_displayed = img_original.copy()
            if cv2.getWindowProperty("Warped Image", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("Warped Image")
            print("Punkte zurückgesetzt. Bitte wählen Sie erneut 4 Punkte aus.")

        elif key == ord("s") or key == ord("S"):
            if cv2.getWindowProperty("Warped Image", cv2.WND_PROP_VISIBLE) > 0:
                if(len(selected_points) == 4):
                    src_pts = order_points(selected_points)
                    dst_pts = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype="float32")
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(img_original, M, (width, height))

                    cv2.imwrite(args.output, warped)
                    print(f"Transformiertes Bild wurde unter '{args.output}' gespeichert.")
                    
            else:
                print("Kein transformiertes Bild zum Speichern verfügbar.")

        elif key == ord("q") or key == ord("Q"):
            print("Programm wird beendet.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":    main()