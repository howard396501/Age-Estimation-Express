/*#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
	//create the cascade classifier object used for the face detection
	CascadeClassifier face_cascade;
	//use the hearcascade_frontalface_alt.xml library
	face_cascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");

	//setup video capture device and link it to the first capture device
	VideoCapture captureDevice;
	captureDevice.open(0);

	//setup image files used is the capture process
	Mat captureFrame;
	Mat grayscaleFrame;

	//create a window to present the results
	namedWindow("outputCapture", 1);

	//create a loop to capture and find faces
	while (true)
	{
		//capture a new image frame
		captureDevice >> captureFrame;

		//convert captured image to gray scale and equalize
		cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
		equalizeHist(grayscaleFrame, grayscaleFrame);

		//create a vector array to store the face found
		std::vector<Rect> faces;

		//find faces and store them in the vector array
		face_cascade.detectMultiScale(grayscaleFrame, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_SCALE_IMAGE, Size(20, 20));

		//draw a rectangle for all found faces in the vector array on the original image
		for (int i = 0; i < faces.size(); i++)
		{
			Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			Point pt2(faces[i].x, faces[i].y);

			rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 0, 0), 1, 8, 0);

			Point pt3(faces[i].x + faces[i].width, faces[i].y + faces[i].height / 4);
			Point pt4(faces[i].x, faces[i].y);

			rectangle(captureFrame, pt3, pt4, cvScalar(255, 0, 0, 0), 1, 8, 0);
		}



		//print the output
		imshow("outputCapture", captureFrame);

		//pause for 33ms
		waitKey(33);
	}

	return 0;
}*/
/*#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	argv[1] = "D:\\1.jpg";

	Mat image;

	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);//載入圖檔

	namedWindow("Display Window", CV_WINDOW_NORMAL);//建立顯示圖檔視窗

	imshow("Display Window", image);//在視窗內顯示圖檔

	waitKey(0);//視窗等待按鍵

	return 0;
}*/

/*-------------------------------------------------人臉偵測*/
/*
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include <stdio.h>

#include <opencv2\opencv.hpp>
#include <cstdio>

using namespace std;
using namespace cv;
void detectAndDisplay(Mat frame);

String face_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

String window_name = "臉部偵測";

int main(void)
{
	VideoCapture capture;
	Mat frame;

	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!) 無法載入臉部 cascade\n");
		return -1;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("--(!) 無法載入眼睛 cascade\n");
		return -1;
	}

	//開啟攝影機並讀取攝影影像
	capture.open("D:\\100450451.jpg");
	if (!capture.isOpened())
	{
		printf("--(!) 無法開啟攝影機\n");
		getchar();
		return -1;
	}

	//讀取影格
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf("--(!) 無法讀取影格\n");
			getchar();
			break;
		}

		//用階梯式類別 進行攝影影格處理
		detectAndDisplay(frame);

		int c = waitKey(10);
		if ((char)c == 27)
		{
			break; //ESC鍵
		}
	}
	return 0;
}
*/

//函式 detectAndDisplay
/*
void detectAndDisplay(Mat frame)
{
	vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY); //將影格轉成灰階
	equalizeHist(frame_gray, frame_gray);//灰階影格進行"直條式"的評等化

	//偵測臉部
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);

		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		
		printf("繪製臉部, 臉部的大小 %d..................\n", faces.size());

		Mat faceROI = frame_gray(faces[i]);
		vector<Rect> eyes;

		//在臉部偵測眼睛
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);

			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);

			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
			//printf("繪製眼睛, 眼睛數量 %d\n\n\n", eyes.size());
		}

		//在臉部偵測額頭
		//vector<Rect> title;

		//for (size_t k = 0; k < title.size(); k++)
		//{
		//	Point title_center(faces[i].x + title[k].width / 2, faces[i].y + title[k].height / 2);
		//	int radius = cvRound((title[k].width + title[k].height)*0.5);

		//	circle(frame, title_center, radius, Scalar(255, 255, 0), 4, 4, 0);
		//}
	}
	// 顯示結果
	imshow(window_name, frame);
}
*/
/*-------------------------------------------------人臉偵測*/


/*
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;


String face_cascade_name = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

int main()
{
	char file[] = "D:\\100450451.jpg";
	
	IplImage *image;
	image = cvLoadImage(file, CV_LOAD_IMAGE_GRAYSCALE);//讀取圖片並轉成灰階
	cvThreshold(image, image, 128, 255, CV_THRESH_BINARY);
	cvShowImage("image", image);
	cvWaitKey(0);
}
*/

#include <opencv\highgui.h>
#include <opencv\cv.h>
#include <opencv2\opencv.hpp>
#include <stdlib.h>  
#include <stdio.h> 
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;


int main()
{
	
	IplImage * img = cvLoadImage("D:\\GF.jpg");
	
	//cvShowImage("age estimation test", img);→用不太到，必要時拿來顯示用

	IplImage * gery = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);//創建一個灰階圖像，1是單通道，所以是灰階
	cvCvtColor(img, img, CV_BGR2YCrCb);//將img轉成img，從BGR轉成YCrCb
	cvInRangeS(img, cvScalar(0, 137, 77), cvScalar(256, 177, 127), gery);//找特定顏色，找到就顯示白色，沒找到就顯示黑色
	cvThreshold(gery, gery, 120, 255, THRESH_BINARY);//Yo~~這是二值化
	cvErode(gery, gery, 0, 1); //浸蝕
	cvDilate(gery, gery, 0, 1); //再膨脹
	//上面是"斷開"之動作
	//cvShowImage("age estimation test gery", gery); //→ cv的show圖片，可能會轉換格式

	Mat mg = cvarrToMat(gery, true);//型態轉換IplImage轉Mat
	imshow("age estimation test gery", mg);

	//-------------以下是找出膚色
	CvMemStorage * storage = cvCreateMemStorage(0);//用來儲存陣列、圖形的動態資料結構：blog.csdn.net/delltdk/article/details/23260993
	//繼續上面的註解：括號裡面是block size，大小是0就預設容量是64KB
	CvSeq * contour = 0;//CvSeq是可增長的陣列，不是固定，很屌；設為0是不會被傳遞給任何使用特定陣列的函數
	CvSeq * contmax = 0;
	CvRect rect = cvRect(0, 0, 0, 0);//CvRect是感興趣區域(ROI)的專用資料結構，而cvRect是畫方框
	//cvRect資料結構：(x值，y值，width，height)
	cvFindContours(gery, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//這是cvFindContours的註解：這是找出輪廓(Contour)，gery是灰階後二值化之圖(也可以直接二值化)，找出其白色之地方
	double area, maxArea = 100;
	for (; contour != 0; contour = contour->h_next)//h_next指向CvSeq，指標結構
	{
		area = fabs(cvContourArea(contour, CV_WHOLE_SEQ));//cvContourArea計算整個或部分輪廓之面積
		//延續上面註解：contour是頂點的組合，後面的是"slice"，是感興趣輪廓的起點和終點，預設是計算整個輪廓面積LAAAAAAAA~~
		//printf("area == %lf\n", area);
		if (area > maxArea)
		{
			contmax = contour;
			maxArea = area;
			rect = cvBoundingRect(contmax, 0);//計算點集的最外面矩形邊界，0是不計算矩形邊界，直接由輪廓rect區域取得該值
			//printf("x=%d,y=%d,width=%d,height=%d", rect.x, rect.y, rect.width, rect.height);
		}
	}
	//-------------下面是邊緣偵測 + 中值濾波 + 特徵膨脹 + 特徵擷取-------------
	Mat mi = cvarrToMat(img, true);//型態轉換IplImage轉Mat
	GaussianBlur(mi, mi, Size(3, 3), 0, 0);
	GaussianBlur(mg, mg, Size(3, 3), 0, 0);
	Mat midst, mgdst;
	Mat midst2, mgdst2;
	//Mat mgdst3;//霍夫曼線的測試
	Canny(mi, midst, 10, 50, 3); 
	Canny(mg, mgdst, 10, 50, 3);//邊緣偵測→需要再調整！！！

	//cvtColor(midst, mgdst3, CV_GRAY2BGR);//霍夫曼線的測試

	medianBlur(midst, midst2, 1);
	medianBlur(mgdst, mgdst2, 1);//中值濾波→需要再調整！！！只有奇數可以調整強度！！

	IplImage * midst2trans;
	midst2trans = &IplImage(midst2);
	cvDilate(midst2trans, midst2trans, 0, 1); //膨脹，特徵之膨脹哦哦哦哦哦哦哦哦～～～～～～～～～

	/*Mat midst3 = cvarrToMat(midst2trans, true);
	std::vector<KeyPoint> keyPoints;
	FastFeatureDetector fast(20);                       FastFeatureDetector偵測！本來是用來判別白線，但似乎沒用了。
	fast.detect(midst3, keyPoints);
	drawKeypoints(midst3, keyPoints, midst3, Scalar::all(255), DrawMatchesFlags::DRAW_OVER_OUTIMG);*/

	//bilateralFilter(mgdst, mgdst2, 1, 30, 30);
	imshow("age estimation test cannyedge_color", midst);//這條code沒用，測試用！→有用！！
	imshow("age estimation test cannyedge_medianBlur_color", midst2);//這條code沒用，測試用！→有用！！
	//imshow("age estination test cannyedge_gery", mgdst);
	//imshow("age estination test cannyedge_medianBlur_gery", mgdst2);
	cvShowImage("age estimation test Dilate", midst2trans);

	CvRect rect2 = cvRect(rect.x+100, rect.y+50, rect.x + rect.width/2-200, rect.y + rect.height/5 -50);//+200 +0 -100   +10
	cvSetImageROI(midst2trans, rect2); //特徵擷取囉！
	CvSize Size;
	Size = cvGetSize(midst2trans);
	cvShowImage("age estimation test ROI", midst2trans);

	IplImage* color_dst;
	CvMemStorage* storage2 = cvCreateMemStorage(0);
	CvSeq* lines = 0;
	int i;

	color_dst = cvCreateImage(cvGetSize(midst2trans), 8, 3);
	cvCvtColor(midst2trans, color_dst, CV_GRAY2BGR);
	lines = cvHoughLines2(midst2trans, storage2, CV_HOUGH_STANDARD, 1, CV_PI / 20, 500, 0, 0);
	for (i = 0; i < MIN(lines->total, 100); i++)
	{
		float * line = (float * ) cvGetSeqElem(lines, i);
		float rho = line[0];
		float theta = line[1];
		CvPoint pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		cvLine(color_dst, pt1, pt2, CV_RGB(255, 0, 0), 2, CV_AA, 0);
	}
	//printf("\n\n%d\n\n", i);
	lines = cvHoughLines2(midst2trans, storage2, CV_HOUGH_PROBABILISTIC, 1, CV_PI / 180, 50, 50, 10);
	for (i = 0; i < lines->total; i++)
	{
		CvPoint* line = (CvPoint*)cvGetSeqElem(lines, i);
		cvLine(color_dst, line[0], line[1], CV_RGB(255, 0, 0), 3, CV_AA, 0);
	}

	if (i >= 0 && i <= 10)
	{
		int a = 1;
		printf("\nforehead wrinkles = %d\n", a);
		printf("Age is 0~15");
	}

	if (i >= 11 && i <= 20)
	{
		int b = 2;
		printf("\nforehead wrinkles = %d\n", b);
		printf("Age is 16~29");
	}

	if (i >= 21 && i <= 30)
	{
		int c = 3;
		printf("\nforehead wrinkles = %d\n", c);
		printf("Age is 30~59");
	}

	if (i >= 31 && i <= 40)
	{
		int d = 4;
		printf("\nforehead wrinkles = %d\n", d);
		printf("Age is 30~59");
	}

	if (i >= 41 && i <= 50)
	{
		int g = 5;
		printf("\nforehead wrinkles = %d\n", g);
		printf("Age is 30~59");
	}

	if (i >= 51 && i <= 60)
	{
		int h = 6;
		printf("\nforehead wrinkles = %d\n", h);
		printf("Age is 60~79");
	}

	if (i >= 61 && i <= 70)
	{
		int w = 7;
		printf("\nforehead wrinkles = %d\n", w);
		printf("Age is 80+");
	}

	if (i >= 71 && i <= 80)
	{
		int s =8;
		printf("\nforehead wrinkles = %d\n", s);
		printf("Age is 80+");
	}

	if (i >= 71 && i <= 80)
	{
		int s = 8;
		printf("\nforehead wrinkles = %d\n", s);
		printf("Age is 80+");
	}

	if (i >= 81 && i <= 90)
	{
		int o = 9;
		printf("\nforehead wrinkles = %d\n", o);
		printf("Age is 80+");
	}

	if (i >= 91 && i <= 100)
	{
		int p = 10;
		printf("\nforehead wrinkles = %d\n", p);
		printf("Age is 80+");
	}
	//printf("\n\n%d\n\n", i);
	cvNamedWindow("Hough", 1);
	cvShowImage("Hough", color_dst);
	/*Mat midst3 = cvarrToMat(midst2trans, true);
	Mat miidst3;
	cvtColor(midst3, miidst3, CV_GRAY2BGR);
	vector<Vec2f> lines;
	HoughLines(midst3, lines, CV_PI / 180, 150, 0, 0);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(miidst3, pt1, pt2, Scalar(255, 0, 0), 1, CV_AA);
	}
	imshow("age estimation test HoughLines", miidst3);*/

	//-------------上面是邊緣偵測 + 中值濾波 + 特徵膨脹 + 特徵擷取-------------
	cvRectangle(img, cvPoint(rect.x, rect.y),
		cvPoint(rect.x + rect.width, rect.y + rect.height),
		cvScalar(0, 255, 0), 2, 8, 0); //這是眶出臉！
	cvRectangle(img, cvPoint(rect.x+150, rect.y),
		cvPoint(rect.x + rect.width/2+150, rect.y + rect.height/5 + 10 ),
		cvScalar(0, 0, 255), 2, 8, 0);//土法煉鋼的眶抬頭紋
	cvCvtColor(img, img, CV_YCrCb2BGR); //YCrCb轉回BGR，不然顏色會很機掰
	cvShowImage("age estimation test", img);
	//-------------以上是找出膚色
	cvWaitKey(0);
	//-------------下面是ROI-------------
	/*
	IplImage * mgdst2trans;//型態轉換Mat轉IplImage，轉邊緣偵測和中值濾波後的灰階圖
	mgdst2trans = &IplImage(midst);//型態轉換Mat轉IplImage，轉邊緣偵測和中值濾波後的灰階圖

	CvRect rect2 = cvRect(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
	cvSetImageROI(mgdst2trans, rect2);
	CvSize Size;
	Size = cvGetSize(mgdst2trans);
	cvShowImage("age estimation test ROI", mgdst2trans);*/
	//-------------上面是ROI-------------
								   
	/*---------------以下是8連通，必要時使用
	Mat labelImage(m.size(), CV_32S);
	int nLabels = connectedComponents(m, labelImage, 8);
	imshow("age estimation test", labelImage);
	---------------以上是8連通，必要時使用*/
	
	/*---浸蝕膨脹殘留code
	Mat src = imread("D:\\100450451.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2;
	Mat src3;
	threshold(src, src3, 120, 255, THRESH_BINARY);	
	Mat dst1;
	Mat dst2;
	erode(src3, dst1, Mat());
	dilate(dst1, dst2, Mat());
	imshow("二值化", src3);
	imshow("浸蝕", dst1);
	imshow("浸蝕後膨脹", dst2);
	waitKey(0);
	return 0;
	浸蝕膨脹殘留code---*/
	
}

