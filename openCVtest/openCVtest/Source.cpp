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

	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);//���J����

	namedWindow("Display Window", CV_WINDOW_NORMAL);//�إ���ܹ��ɵ���

	imshow("Display Window", image);//�b��������ܹ���

	waitKey(0);//�������ݫ���

	return 0;
}*/

/*-------------------------------------------------�H�y����*/
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

String window_name = "�y������";

int main(void)
{
	VideoCapture capture;
	Mat frame;

	if (!face_cascade.load(face_cascade_name))
	{
		printf("--(!) �L�k���J�y�� cascade\n");
		return -1;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		printf("--(!) �L�k���J���� cascade\n");
		return -1;
	}

	//�}����v����Ū����v�v��
	capture.open("D:\\100450451.jpg");
	if (!capture.isOpened())
	{
		printf("--(!) �L�k�}����v��\n");
		getchar();
		return -1;
	}

	//Ū���v��
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf("--(!) �L�kŪ���v��\n");
			getchar();
			break;
		}

		//�ζ��覡���O �i����v�v��B�z
		detectAndDisplay(frame);

		int c = waitKey(10);
		if ((char)c == 27)
		{
			break; //ESC��
		}
	}
	return 0;
}
*/

//�禡 detectAndDisplay
/*
void detectAndDisplay(Mat frame)
{
	vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY); //�N�v���ন�Ƕ�
	equalizeHist(frame_gray, frame_gray);//�Ƕ��v��i��"������"��������

	//�����y��
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);

		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		
		printf("ø�s�y��, �y�����j�p %d..................\n", faces.size());

		Mat faceROI = frame_gray(faces[i]);
		vector<Rect> eyes;

		//�b�y����������
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);

			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);

			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
			//printf("ø�s����, �����ƶq %d\n\n\n", eyes.size());
		}

		//�b�y�������B�Y
		//vector<Rect> title;

		//for (size_t k = 0; k < title.size(); k++)
		//{
		//	Point title_center(faces[i].x + title[k].width / 2, faces[i].y + title[k].height / 2);
		//	int radius = cvRound((title[k].width + title[k].height)*0.5);

		//	circle(frame, title_center, radius, Scalar(255, 255, 0), 4, 4, 0);
		//}
	}
	// ��ܵ��G
	imshow(window_name, frame);
}
*/
/*-------------------------------------------------�H�y����*/


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
	image = cvLoadImage(file, CV_LOAD_IMAGE_GRAYSCALE);//Ū���Ϥ����ন�Ƕ�
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
	
	//cvShowImage("age estimation test", img);���Τ��Ө�A���n�ɮ�����ܥ�

	IplImage * gery = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);//�Ыؤ@�ӦǶ��Ϲ��A1�O��q�D�A�ҥH�O�Ƕ�
	cvCvtColor(img, img, CV_BGR2YCrCb);//�Nimg�নimg�A�qBGR�নYCrCb
	cvInRangeS(img, cvScalar(0, 137, 77), cvScalar(256, 177, 127), gery);//��S�w�C��A���N��ܥզ�A�S���N��ܶ¦�
	cvThreshold(gery, gery, 120, 255, THRESH_BINARY);//Yo~~�o�O�G�Ȥ�
	cvErode(gery, gery, 0, 1); //���k
	cvDilate(gery, gery, 0, 1); //�A����
	//�W���O"�_�}"���ʧ@
	//cvShowImage("age estimation test gery", gery); //�� cv��show�Ϥ��A�i��|�ഫ�榡

	Mat mg = cvarrToMat(gery, true);//���A�ഫIplImage��Mat
	imshow("age estimation test gery", mg);

	//-------------�H�U�O��X����
	CvMemStorage * storage = cvCreateMemStorage(0);//�Ψ��x�s�}�C�B�ϧΪ��ʺA��Ƶ��c�Gblog.csdn.net/delltdk/article/details/23260993
	//�~��W�������ѡG�A���̭��Oblock size�A�j�p�O0�N�w�]�e�q�O64KB
	CvSeq * contour = 0;//CvSeq�O�i�W�����}�C�A���O�T�w�A���x�F�]��0�O���|�Q�ǻ�������ϥίS�w�}�C�����
	CvSeq * contmax = 0;
	CvRect rect = cvRect(0, 0, 0, 0);//CvRect�O�P����ϰ�(ROI)���M�θ�Ƶ��c�A��cvRect�O�e���
	//cvRect��Ƶ��c�G(x�ȡAy�ȡAwidth�Aheight)
	cvFindContours(gery, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//�o�OcvFindContours�����ѡG�o�O��X����(Contour)�Agery�O�Ƕ���G�ȤƤ���(�]�i�H�����G�Ȥ�)�A��X��զ⤧�a��
	double area, maxArea = 100;
	for (; contour != 0; contour = contour->h_next)//h_next���VCvSeq�A���е��c
	{
		area = fabs(cvContourArea(contour, CV_WHOLE_SEQ));//cvContourArea�p���өγ������������n
		//����W�����ѡGcontour�O���I���զX�A�᭱���O"slice"�A�O�P����������_�I�M���I�A�w�]�O�p���ӽ������nLAAAAAAAA~~
		//printf("area == %lf\n", area);
		if (area > maxArea)
		{
			contmax = contour;
			maxArea = area;
			rect = cvBoundingRect(contmax, 0);//�p���I�����̥~���x����ɡA0�O���p��x����ɡA�����ѽ���rect�ϰ���o�ӭ�
			//printf("x=%d,y=%d,width=%d,height=%d", rect.x, rect.y, rect.width, rect.height);
		}
	}
	//-------------�U���O��t���� + �����o�i + �S�x���� + �S�x�^��-------------
	Mat mi = cvarrToMat(img, true);//���A�ഫIplImage��Mat
	GaussianBlur(mi, mi, Size(3, 3), 0, 0);
	GaussianBlur(mg, mg, Size(3, 3), 0, 0);
	Mat midst, mgdst;
	Mat midst2, mgdst2;
	//Mat mgdst3;//�N�Ұҽu������
	Canny(mi, midst, 10, 50, 3); 
	Canny(mg, mgdst, 10, 50, 3);//��t�������ݭn�A�վ�I�I�I

	//cvtColor(midst, mgdst3, CV_GRAY2BGR);//�N�Ұҽu������

	medianBlur(midst, midst2, 1);
	medianBlur(mgdst, mgdst2, 1);//�����o�i���ݭn�A�վ�I�I�I�u���_�ƥi�H�վ�j�סI�I

	IplImage * midst2trans;
	midst2trans = &IplImage(midst2);
	cvDilate(midst2trans, midst2trans, 0, 1); //���ȡA�S�x�����Ȯ@�@�@�@�@�@�@�@����������

	/*Mat midst3 = cvarrToMat(midst2trans, true);
	std::vector<KeyPoint> keyPoints;
	FastFeatureDetector fast(20);                       FastFeatureDetector�����I���ӬO�ΨӧP�O�սu�A�����G�S�ΤF�C
	fast.detect(midst3, keyPoints);
	drawKeypoints(midst3, keyPoints, midst3, Scalar::all(255), DrawMatchesFlags::DRAW_OVER_OUTIMG);*/

	//bilateralFilter(mgdst, mgdst2, 1, 30, 30);
	imshow("age estimation test cannyedge_color", midst);//�o��code�S�ΡA���եΡI�����ΡI�I
	imshow("age estimation test cannyedge_medianBlur_color", midst2);//�o��code�S�ΡA���եΡI�����ΡI�I
	//imshow("age estination test cannyedge_gery", mgdst);
	//imshow("age estination test cannyedge_medianBlur_gery", mgdst2);
	cvShowImage("age estimation test Dilate", midst2trans);

	CvRect rect2 = cvRect(rect.x+100, rect.y+50, rect.x + rect.width/2-200, rect.y + rect.height/5 -50);//+200 +0 -100   +10
	cvSetImageROI(midst2trans, rect2); //�S�x�^���o�I
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

	//-------------�W���O��t���� + �����o�i + �S�x���� + �S�x�^��-------------
	cvRectangle(img, cvPoint(rect.x, rect.y),
		cvPoint(rect.x + rect.width, rect.y + rect.height),
		cvScalar(0, 255, 0), 2, 8, 0); //�o�O���X�y�I
	cvRectangle(img, cvPoint(rect.x+150, rect.y),
		cvPoint(rect.x + rect.width/2+150, rect.y + rect.height/5 + 10 ),
		cvScalar(0, 0, 255), 2, 8, 0);//�g�k�ҿ��������Y��
	cvCvtColor(img, img, CV_YCrCb2BGR); //YCrCb��^BGR�A���M�C��|�ܾ��T
	cvShowImage("age estimation test", img);
	//-------------�H�W�O��X����
	cvWaitKey(0);
	//-------------�U���OROI-------------
	/*
	IplImage * mgdst2trans;//���A�ഫMat��IplImage�A����t�����M�����o�i�᪺�Ƕ���
	mgdst2trans = &IplImage(midst);//���A�ഫMat��IplImage�A����t�����M�����o�i�᪺�Ƕ���

	CvRect rect2 = cvRect(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
	cvSetImageROI(mgdst2trans, rect2);
	CvSize Size;
	Size = cvGetSize(mgdst2trans);
	cvShowImage("age estimation test ROI", mgdst2trans);*/
	//-------------�W���OROI-------------
								   
	/*---------------�H�U�O8�s�q�A���n�ɨϥ�
	Mat labelImage(m.size(), CV_32S);
	int nLabels = connectedComponents(m, labelImage, 8);
	imshow("age estimation test", labelImage);
	---------------�H�W�O8�s�q�A���n�ɨϥ�*/
	
	/*---���k���ȴݯdcode
	Mat src = imread("D:\\100450451.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2;
	Mat src3;
	threshold(src, src3, 120, 255, THRESH_BINARY);	
	Mat dst1;
	Mat dst2;
	erode(src3, dst1, Mat());
	dilate(dst1, dst2, Mat());
	imshow("�G�Ȥ�", src3);
	imshow("���k", dst1);
	imshow("���k�῱��", dst2);
	waitKey(0);
	return 0;
	���k���ȴݯdcode---*/
	
}

