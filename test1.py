import cv2
import numpy as np

#画像の読み込み r…エスケープシーケンス回避
str = r"C:\Users\start\mysite\milkdrop.bmp"
img = cv2.imread(str)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#読み込み確認 画像の表示
#cv2.imshow("milkdrop", img)

#画像フィルタ 輪郭線の強調 パラメータ選定は手探り。
img_fil = cv2.bilateralFilter(img, 10, 10, 7)

#画像のグレースケール化
grayImg = cv2.cvtColor(img_fil, cv2.COLOR_BGR2GRAY)

#2値画像の生成、閾値取得
ret, binaryImg = cv2.threshold(grayImg, 130, 255, cv2.THRESH_OTSU)

#輪郭線の抽出・選別 パラメータ生成は手探り。
contours, hierarchy = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE ) 
contours_by_area = list(filter(lambda x: cv2.contourArea(x) >= 1500, contours))

#マスクサイズ確認用
#for i, contour in enumerate(contours_by_area):
#    print(cv2.contourArea(contour))

# マスク画像を作成する。
mask = np.full(img.shape[:2], 0, dtype=img.dtype)

#マスクを合成 shapeを合わせる
mask = cv2.drawContours(mask, contours_by_area, -1, color=255, thickness=-1)
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
#マスクの確認
#cv2.imshow("mask",mask)

#結果画像の表示
dst = cv2.bitwise_and(img, mask)
cv2.imshow("result",dst)

cv2.waitKey()
cv2.destroyAllWindows()