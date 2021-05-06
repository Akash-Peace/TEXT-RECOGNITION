import cv2, numpy as np, pytesseract, enchant
from imutils.object_detection import non_max_suppression
def start():
    cap = cv2.VideoCapture(0)
    file = open("text_recognition_matched_list.txt", "a")
    acc = []
    while True:
        ret, orig_img = cap.read()
        h, w, _ = orig_img.shape
        resizing = 320  # EAST algo needs image density multiplies by 32. So, here we using overall size(w*h) 320 as a standard.
        img = cv2.resize(orig_img, (resizing, resizing))
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        net = cv2.dnn.readNet('frozen_east_text_detection.pb')
        blob = cv2.dnn.blobFromImage(img, 1.0, (resizing, resizing), (123.68, 116.78, 103.94), True, False)
        net.setInput(blob)
        scores, geometry = net.forward(layerNames)
        _, _, rows, cols = scores.shape
        recs, conf = [], []
        # PREDICTION: Calculates the probability of prediction in each row & column.
        for y in range(rows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(cols):
                if scoresData[x] < 0.5:
                    continue
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                dh = xData0[x] + xData2[x]
                dw = xData1[x] + xData3[x]
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - dw)
                startY = int(endY - dh)
                recs.append((startX, startY, endX, endY))
                conf.append(scoresData[x])
        # PREDICTION: Ends here.
        boxes = non_max_suppression(np.array(recs), probs=conf)  # To reduce overlapping.
        rH, rW = h / resizing, w / resizing  # values for recovering positions to original image.
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            '''# This section is for padding, where if result is incorrect, make dx,dy's 0 to 0.05 or 0.10.
            dX = int((endX - startX) * 0)
            dY = int((endY - startY) * 0)
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(w, endX + (dX * 2))
            endY = min(h, endY + (dY * 2))
            '''
            roi = orig_img[startY:endY, startX:endX]  # Cropping
            config = ("-l eng --oem 1 --psm 7")  # key configuration for pytesseract
            try:
                text = pytesseract.image_to_string(roi, config=config)
                text = "".join([char for char in text if char.isalnum()]).strip()  # For stripping non-ascii values
                if enchant.Dict("en_US").check(text):   # and (text not in file.read()): not working :(.
                    if text not in acc:
                        acc.insert(0, text)
                        file.write(text)
                        file.write("\n")
                    bgr = (0, 255, 0)
                else:
                    bgr = (0, 0, 255)
                    text = f'May be "{str(*(enchant.Dict("en_US").suggest(text))[:1])}"'
                cv2.rectangle(orig_img, (startX, startY), (endX, endY), bgr, 2)
                cv2.putText(orig_img, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, bgr, 2)
            except Exception:
                pass
        cv2.putText(orig_img, f'100% matched : {", ".join(acc[:4])} ...', (25, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.namedWindow("Text Recognition - Press q to exit", cv2.WINDOW_NORMAL)
        cv2.imshow("Text Recognition - Press q to exit", orig_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    file.close()
    cap.release()
    cv2.destroyAllWindows()
