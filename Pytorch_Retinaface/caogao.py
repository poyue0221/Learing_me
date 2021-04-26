import cv2


img = cv2.imread("./widerface_test/量化前图片/1.jpg")

img_rw = cv2.resize(img, (1280,720))

print(img.shape)
print(img_rw.shape)
print(int((1280/1024)*100))
print(int((720/713)*80))
# cv2.circle(img_rw, (int((640/1024)*100), int((320/713)*80)), 1, (0, 0, 255), 4)
cv2.rectangle(img_rw, (int((1280/1024)*100), int((720/713)*80)), (int((1280/1024)*100+20), int((720/713)*80)+20), (0, 0, 255), 2)
cv2.imshow("TEST", img)
cv2.waitKey(100000)