// g++ main.cpp -lm -lgsl -lgslcblas  `pkg-config --cflags --libs opencv` -lpthread -O2 -Wall

#include <stdlib.h>
#include <math.h>
#include "cv.h"
#include "highgui.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit.h>


using namespace std;
using namespace cv;


Point2f pt_1, pt_2; // some point
bool addRemovePt_1 = false, addRemovePt_2 = false; // flags
int number_of_points = 0; // some ints
const double camera_param_1[12] = {1030.98, 0, 335.175, 0, 0, 1019.4, 163.034, 0, 0, 0, 1, 0};
const double camera_param_2[12] = {1069.45, 94.0257, -104.689, 12728.5, -27.5149, 1016.57, 168.52, 31.615, 0.386178, 0.008008, 0.922389, 2.85243};
double r[30] = {0.0}; // the computed 3D position from stereo vision


void triangulation(double imgA_x, double imgA_y, double imgB_x, double imgB_y, double *p_x, double *p_y, double *p_z); // stereo triangulation function


// mouse handler 1
void onMouse_1( int event, int X, int Y, int flags, void* ) {
	if ( event == CV_EVENT_LBUTTONDOWN && (number_of_points < 2) ) { 
		number_of_points++; // increment counter
		addRemovePt_1 = true; // used for adding more tracking points
		pt_1 = Point2f ((float)X,(float)Y); // get track point
	}
} // end of mouse handler

// mouse handler 2
void onMouse_2( int event, int X, int Y, int flags, void* ) {
	if ( event == CV_EVENT_LBUTTONDOWN && (number_of_points >= 2 && number_of_points < 4) ) {
    number_of_points++; // increment counter
    addRemovePt_2 = true; // used for adding more tracking points
    pt_2 = Point2f ((float)X,(float)Y); // get track point
  }
} // end of mouse handler


int main(void)
{
	// local variables
	Mat gray_1, prevGray_1, image_1, gray_2, prevGray_2, image_2; // images
	vector<Point2f> points_1[2], points_2[2]; // points array to be used within the tracking algorithm
	VideoCapture cap_1, cap_2; // the video capture structure
	TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03); // when to terminate
	Size winSize(10,10); // size of the tracking window
	const int MAX_COUNT_1 = 2, MAX_COUNT_2 = 2; // number of points to track per camera
	bool VIDEO_RECORD = false, VIDEO_FILE_OK = false; // flag to control opening video file
	VideoWriter record_1, record_2; // to record video from camera
	char c = 0; // get pressed key
	Point2f point_1A, point_1B; // image points
	Point2f point_2A, point_2B; // image points
	double distance = 0.0; // distance between feature points

	// initialisation
  namedWindow( "Cam1", 0 ); cvMoveWindow("Cam1", 375, 0); // create window
  namedWindow( "Cam2", 0 ); cvMoveWindow("Cam2", 1200, 0); // create window
  setMouseCallback( "Cam1", onMouse_1, 0 );
  setMouseCallback( "Cam2", onMouse_2, 0 );
  cap_1.open(0); cap_2.open(1); if( !cap_1.isOpened() || !cap_2.isOpened() ) printf("ERROR: Camera bu hao\7\n");
  
	// loop
	for(;;)
	{
  	// get a new frame from camera
		Mat frame; // create a new frame
  	
  	// get image 1
		cap_1 >> frame; // capture current image
		if( frame.empty() ){ printf("\n\nERROR: cannot read camera \n\n"); exit(1);	}
		frame.copyTo(image_1); // copy same data to the image_1 structure
		cvtColor(image_1, gray_1, CV_BGR2GRAY); // and cast it in gray scale

		// get image 2
		cap_2 >> frame; // capture current image
		if( frame.empty() ){ printf("\n\nERROR: cannot read camera \n\n"); exit(1);	}
		frame.copyTo(image_2); // copy same data to the image_1 structure
		cvtColor(image_2, gray_2, CV_BGR2GRAY); // and cast it in gray scale

		// tracking image 1
		if( !points_1[0].empty() ) // once points defined, do
		{ 
			vector<uchar> status;
			vector<float> err;
			if(prevGray_1.empty()) gray_1.copyTo(prevGray_1); // initialise previous gray image
			// the tracking algorithm
			calcOpticalFlowPyrLK(prevGray_1,gray_1,points_1[0],points_1[1],status,err,winSize,3,termcrit,0);
			size_t i, k;
			for( i = k = 0; i < points_1[1].size(); i++ ) { // for all points do
				if( addRemovePt_1 ) // add points that are not too close
					if( norm(pt_1 - points_1[1][i]) <= 5 ) {
						addRemovePt_1 = false; continue;
					}	
				if( !status[i] ) continue;
				points_1[1][k++] = points_1[1][i];
				circle( image_1, points_1[1][i], 2, Scalar(0,0,255), 5, 8); // draw points
			}
			points_1[1].resize(k);
		} // end of tracking image 1
		
		// tracking image 2
		if( !points_2[0].empty() ) // once points defined, do
		{ 
			vector<uchar> status;
			vector<float> err;
			if(prevGray_2.empty()) gray_2.copyTo(prevGray_2); // initialise previous gray image
			// the tracking algorithm
			calcOpticalFlowPyrLK(prevGray_2,gray_2,points_2[0],points_2[1],status,err,winSize,3,termcrit,0);
			size_t i, k;
			for( i = k = 0; i < points_2[1].size(); i++ ) { // for all points do
				if( addRemovePt_2 ) // add points that are not too close
					if( norm(pt_2 - points_2[1][i]) <= 5 ) {
						addRemovePt_2 = false; continue;
					}	
				if( !status[i] ) continue;
				points_2[1][k++] = points_2[1][i];
				circle( image_2, points_2[1][i], 2, Scalar(0,0,255), 5, 8); // draw points
			}
			points_2[1].resize(k);
		} // end of tracking image 2	

		// add new features to track 1
		if( addRemovePt_1 && points_1[1].size() < (size_t)MAX_COUNT_1 ) { // if available
			vector<Point2f> tmp;
			tmp.push_back(pt_1);
			cornerSubPix( gray_1, tmp, winSize, cvSize(-1,-1), termcrit);
			points_1[1].push_back(tmp[0]);
			addRemovePt_1 = false; // block
		}
		
		// add new features to track 2
		if( addRemovePt_2 && points_2[1].size() < (size_t)MAX_COUNT_2 ) { // if available
			vector<Point2f> tmp;
			tmp.push_back(pt_2);
			cornerSubPix( gray_2, tmp, winSize, cvSize(-1,-1), termcrit);
			points_2[1].push_back(tmp[0]);
			addRemovePt_2 = false; // block
		}

		// remane points
		if (number_of_points == 4) { 
			point_1A.x = points_1[1][0].x; point_1A.y = points_1[1][0].y; 
			point_1B.x = points_1[1][1].x; point_1B.y = points_1[1][1].y; 
			point_2A.x = points_2[1][0].x; point_2A.y = points_2[1][0].y;
			point_2B.x = points_2[1][1].x; point_2B.y = points_2[1][1].y;
		}

		// visually measure 3D positions
		triangulation(point_1A.x, point_1A.y, point_2A.x, point_2A.y, &r[0], &r[1], &r[2]);
		triangulation(point_1B.x, point_1B.y, point_2B.x, point_2B.y, &r[3], &r[4], &r[5]);

		// compute distance
		distance = sqrt( (r[0]-r[3])*(r[0]-r[3]) + (r[1]-r[4])*(r[1]-r[4]) + (r[2]-r[5])*(r[2]-r[5]) );	
	
    // print
		printf("distances d1 = %g, d2 = %g\n",sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]),sqrt(r[3]*r[3]+r[4]*r[4]+r[5]*r[5]));
		printf("total distance %g\n",distance*1);
    printf("position: p1 = [%g %g %g], p2 = [%g %g %g]\n\n",r[0],r[1],r[2],r[3],r[4],r[5]);

		// features
		if (number_of_points == 4) {
			line( image_1, point_1A, point_1B, Scalar(0,0,255), 5, 8 ); // draw line
			line( image_2, point_2A, point_2B, Scalar(0,0,255), 5, 8 ); // draw line
		}
	
		// show images
		imshow("Cam1", image_1);
		imshow("Cam2", image_2); 

		// write video file
    if (VIDEO_RECORD && VIDEO_FILE_OK) { 
    	record_1 << image_1;
    	record_2 << image_2;  
    }
    if (VIDEO_RECORD && !VIDEO_FILE_OK) {
      record_1.open("video_a.avi", CV_FOURCC('D','I','V','X'), 24, image_1.size(), true);
      record_2.open("video_b.avi", CV_FOURCC('D','I','V','X'), 24, image_2.size(), true);  
      VIDEO_FILE_OK = true; // block call
    }
		
		// pressed keys
		if ( c == 27) break; // if esc
		else if ( c == 'v' ) VIDEO_RECORD = true; // video record
		else if ( c == 32 ) {
    	number_of_points = 0; // reset count
    	points_1[0].clear(); points_1[1].clear(); // clear points vectors
    	points_2[0].clear(); points_2[1].clear(); // clear points vectors
		}
	
		// current becomes previous
    std::swap(points_1[1], points_1[0]);
		std::swap(points_2[1], points_2[0]);
		swap(prevGray_1, gray_1);
		swap(prevGray_2, gray_2);

		// wait
    c = (char)waitKey(2);	
	}

	printf("End of program\n");
	return 0;
}


// function to measure from stereo images the 3D position of a point
void triangulation(double imgA_x, double imgA_y, double imgB_x, double imgB_y, double *p_x, double *p_y, double *p_z) {
	// local variables
	double Mtx[6][4] = {{0.0}}; // matrix for triangulation algorithm
	gsl_matrix *A = gsl_matrix_alloc (6, 4); // to store the data
	gsl_matrix *V = gsl_matrix_alloc (4, 4); // singular vectors
	gsl_vector *S = gsl_vector_alloc (4); // singular values
	int i = 0, j = 0; // some indices
	double pos_h[4] = {0.0}; // homogenous position 4-vector
	
	// matrix used for agebraic triangulation... Mtx = Img_points[cross_operator] Calibration_Matrices
	Mtx[0][0] = imgA_y*camera_param_1[8] - camera_param_1[4];
	Mtx[0][1] = imgA_y*camera_param_1[9] - camera_param_1[5];
	Mtx[0][2] = imgA_y*camera_param_1[10] - camera_param_1[6];
	Mtx[0][3] = imgA_y*camera_param_1[11] - camera_param_1[7];
	Mtx[1][0] = camera_param_1[0] - imgA_x*camera_param_1[8];
	Mtx[1][1] = camera_param_1[1] - imgA_x*camera_param_1[9];
	Mtx[1][2] = camera_param_1[2] - imgA_x*camera_param_1[10];
	Mtx[1][3] = camera_param_1[3] - imgA_x*camera_param_1[11];
	Mtx[2][0] = imgA_x*camera_param_1[4] - imgA_y*camera_param_1[0];
	Mtx[2][1] = imgA_x*camera_param_1[5] - imgA_y*camera_param_1[1];
	Mtx[2][2] = imgA_x*camera_param_1[6] - imgA_y*camera_param_1[2];
	Mtx[2][3] = imgA_x*camera_param_1[7] - imgA_y*camera_param_1[3];
	//
	Mtx[3][0] = imgB_y*camera_param_2[8] - camera_param_2[4];
	Mtx[3][1] = imgB_y*camera_param_2[9] - camera_param_2[5];
	Mtx[3][2] = imgB_y*camera_param_2[10] - camera_param_2[6];
	Mtx[3][3] = imgB_y*camera_param_2[11] - camera_param_2[7];
	Mtx[4][0] = camera_param_2[0] - imgB_x*camera_param_2[8];
	Mtx[4][1] = camera_param_2[1] - imgB_x*camera_param_2[9];
	Mtx[4][2] = camera_param_2[2] - imgB_x*camera_param_2[10];
	Mtx[4][3] = camera_param_2[3] - imgB_x*camera_param_2[11];
	Mtx[5][0] = imgB_x*camera_param_2[4] - imgB_y*camera_param_2[0];
	Mtx[5][1] = imgB_x*camera_param_2[5] - imgB_y*camera_param_2[1];
	Mtx[5][2] = imgB_x*camera_param_2[6] - imgB_y*camera_param_2[2];
	Mtx[5][3] = imgB_x*camera_param_2[7] - imgB_y*camera_param_2[3];
	
	// put calibration data into matrix A
	for (i = 0; i < (int)(A->size1); i++) 
		for (j = 0; j < (int)(A->size2); j++)  
			gsl_matrix_set(A, i, j, Mtx[i][j]);
			
	// call (thin) SVD algorithm
	gsl_linalg_SV_decomp_jacobi (A, V, S);
			
	// get homogenous 4-vector
	pos_h[0] = gsl_matrix_get(V,0,3);
	pos_h[1] = gsl_matrix_get(V,1,3);
	pos_h[2] = gsl_matrix_get(V,2,3);
	pos_h[3] = gsl_matrix_get(V,3,3);
	
	// stereo-computed position vector
	*p_x = pos_h[0]/pos_h[3];
	*p_y = pos_h[1]/pos_h[3];
	*p_z = pos_h[2]/pos_h[3];	
}




