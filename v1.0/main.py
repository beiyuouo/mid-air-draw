import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    init = 0
    last_point = 0
    font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体
    size = 0.5  # 设置大小
    width, height = 300, 300  # 设置拍摄窗口大小
    x0, y0 = 100, 100  # 设置选取位置
    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 2)
        roi = binaryMask(img, x0, y0, width, height)
        res = skinMask(roi)
        contours = getContours(res)
        if init == 0:
            img2 = roi.copy()
            img2[:, :, :] = 255
            init = 1

        print(len(contours))
        if len(contours) > 0:
            first = [x[0] for x in contours[0]]
            first = np.array(first[:])
            print(first)
            y_min = roi.shape[1]
            idx = 0
            for i, (x, y) in enumerate(first):
                if y < y_min:
                    y_min = y
                    idx = i
            print(first[idx])
            point = (first[idx][0], first[idx][1])
            cv2.circle(img2, point, 1, (255, 0, 0))
            if last_point != 0:
                cv2.line(img2, point, last_point, (255, 0, 0), 1)
            last_point = point

        # print(img2)
        cv2.drawContours(roi, contours, -1, (0, 255, 0), 2)
        cv2.imshow('capture', img)
        cv2.imshow('roi', roi)
        cv2.imshow('draw', img2)
        k = cv2.waitKey(10)
        if k == 27:
            break


def getContours(img):
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    vaildContours = []
    for cont in contours:
        if cv2.contourArea(cont) > 9000:
            # x,y,w,h = cv2.boundingRect(cont)
            # if h/w >0.75:
            # filter face failed
            vaildContours.append(cv2.convexHull(cont))
            # print(cv2.convexHull(cont))
            # rect = cv2.minAreaRect(cont)
            # box = cv2.cv.BoxPoint(rect)
            # vaildContours.append(np.int0(box))
    return vaildContours


def binaryMask(frame, x0, y0, width, height):
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))  # 画出截取的手势框图
    roi = frame[y0:y0 + height, x0:x0 + width]  # 获取手势框图
    return roi


def HSVBin(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_skin = np.array([100, 50, 0])
    upper_skin = np.array([125, 255, 255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # res = cv2.bitwise_and(img,img,mask=mask)
    return mask


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


if __name__ == '__main__':
    main()
