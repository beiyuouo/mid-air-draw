import cv2
import numpy as np
import copy
import math

# from appscript import app

# Environment:
# hardware：Raspberry Pi 4B
# OS    : Raspbian GNU/Linux 10 (buster)
# python: 3.7.3
# opencv: 4.2.0

# parameters
cap_region_x_begin = 0.6  # start point/total width
cap_region_y_end = 0.6  # start point/total width
threshold = 60  # BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works


def skinMask1(roi):
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # 转换到RGB空间
    (R, G, B) = cv2.split(rgb)  # 获取图像每个像素点的RGB的值，即将一个二维矩阵拆成三个二维矩阵
    skin = np.zeros(R.shape, dtype=np.uint8)  # 掩膜
    (x, y) = R.shape  # 获取图像的像素点的坐标范围
    for i in range(0, x):
        for j in range(0, y):
            # 判断条件，不在肤色范围内则将掩膜设为黑色，即255
            if (abs(R[i][j] - G[i][j]) > 15) and (R[i][j] > G[i][j]) and (R[i][j] > B[i][j]):
                if (R[i][j] > 95) and (G[i][j] > 40) and (B[i][j] > 20) \
                        and (max(R[i][j], G[i][j], B[i][j]) - min(R[i][j], G[i][j], B[i][j]) > 15):
                    skin[i][j] = 255
                elif (R[i][j] > 220) and (G[i][j] > 210) and (B[i][j] > 170):
                    skin[i][j] = 255
    # res = cv2.bitwise_and(roi, roi, mask=skin)  # 图像与运算
    return skin


def skinMask2(roi):
    low = np.array([0, 48, 50])  # 最低阈值
    high = np.array([20, 255, 255])  # 最高阈值
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 转换到HSV空间
    mask = cv2.inRange(hsv, low, high)  # 掩膜，不在范围内的设为255
    # res = cv2.bitwise_and(roi, roi, mask=mask)  # 图像与运算
    return mask


def skinMask3(roi):
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 25), 43, 0, 360, (255, 255, 255), -1)  # 绘制椭圆弧线
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, Cr, Cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    skin = np.zeros(Cr.shape, dtype=np.uint8)  # 掩膜
    (x, y) = Cr.shape
    for i in range(0, x):
        for j in range(0, y):
            if skinCrCbHist[Cr[i][j], Cb[i][j]] > 0:  # 若不在椭圆区间中
                skin[i][j] = 255
    # res = cv2.bitwise_and(roi, roi, mask=skin)
    return skin


def skinMask4(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理
    # res = cv2.bitwise_and(roi, roi, mask=skin)
    return skin


def skinMask5(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    skin = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            # 每个像素点进行判断
            if (cr[i][j] > 130) and (cr[i][j] < 175) and (cb[i][j] > 77) and (cb[i][j] < 127):
                skin[i][j] = 255
    # res = cv2.bitwise_and(roi, roi, mask=skin)
    return skin


def skinMask(roi):
    return skinMask4(roi)


def dis(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def printThreshold(thr):
    print("! Changed threshold to " + str(thr))


def removeBG(frame):
    """
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    """
    res = cv2.bitwise_and(frame, frame, mask=skinMask(frame))
    return res


def calculateFingers(res, drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.line(drawing, far, start, [211, 200, 200], 2)
                    cv2.line(drawing, far, end, [211, 200, 200], 2)
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


# Camera
camera = cv2.VideoCapture(0)
# rt = camera.get(10)
# print(rt)
camera.set(10, 150)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

last_point = 0
init = 0

while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)
    print(frame.shape)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        cv2.imshow('mask', img)
        if init == 0:
            img2 = img.copy()
            img2[:, :] = 255
            init = 1
        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        # cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow('ori', thresh)

        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        drawing = np.zeros(img.shape, np.uint8)
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]

            # print(last_point)
            # print(res)
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            # cv2.drawContours(drawing, [], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal, cnt = calculateFingers(res, drawing)
            if cnt > 2:
                img2[:, :] = 255
            # print(cnt)
            if triggerSwitch is True:
                # if isFinishCal is True and cnt <= 2:
                if isFinishCal is True:
                    print(cnt)
                    # app('System Events').keystroke(' ')  # simulate pressing blank space
            if cnt <= 2:
                first = [x[0] for x in contours[ci]]
                first = np.array(first[:])
                # print(first)
                y_min = frame.shape[1]
                idx = 0
                for i, (x, y) in enumerate(first):
                    if y < y_min:
                        y_min = y
                        idx = i
                # print(first[idx])
                point = (first[idx][0], first[idx][1])
                cv2.circle(img2, point, 3, (255, 0, 0))
                if last_point != 0:
                    # print('????')
                    if dis(last_point, point) < 30:
                        cv2.line(img2, point, last_point, (255, 0, 0), 3)
                last_point = point
            '''
                if cnt > 1:
                    first = [x[0] for x in contours[ci]]
                else:
                    first = [x[0] for x in contours[0]]
                first = [x[0] for x in contours[ci]]
                first = np.array(first[:])
                # print(first)
                y_min = frame.shape[1]
                idx = 0
                for i, (x, y) in enumerate(first):
                    if y < y_min:
                        y_min = y
                        idx = i
                # print(first[idx])
                point = (first[idx][0], first[idx][1])
                cv2.circle(img2, point, 3, (255, 255, 255))
                if last_point != 0:
                    # print('????')
                    cv2.line(img2, point, last_point, (255, 255, 255), 3)
                last_point = point
            '''

        cv2.imshow('output', drawing)
        cv2.imshow('draw', img2)

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print('!!!Trigger On!!!')
    elif k == ord('c'):
        img2[:, :] = 255
        print('!!!img2 Clear!!!')
