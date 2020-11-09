import cv2 as cv

src = cv.imread('1.jpg.jpg')
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
ret,binary = cv.threshold(gray, 0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU)  # 像素值小于127全变成0，大于127变成255
cv.imwrite('21.jpg',binary)


# cv.imshow("input image", src)
# # cv.imshow('output',des)
# cv.waitKey(0)
# cv.destroyAllWindows()
