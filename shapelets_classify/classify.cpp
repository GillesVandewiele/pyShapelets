/******************************************************************************
*******************************************************************************
******                                                                  *******
******     This code is written by Abdullah Al Mueen at the department  *******
******     of Computer Science and Engineering of University of         *******
******     California - RIverside.                                      *******
******                                                                  *******
*******************************************************************************
******************************************************************************/

/*#############################################################################
######                                                                  #######
######     This code is open to use, reuse and distribute at the user's #######
######     own risk and as long as the above credit is ON. The complete #######
######     description of the algorithm and methods applied can be      #######
######     found in the paper                                           #######
######                                                                  #######
#############################################################################*/


#include <iostream>
#include <fstream>
#include <sstream>


#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <signal.h>
using namespace std;

#define INF 9999999999999.99
#define DEBUG 1
#define RESULT 1


#include "timeSeries.h"
#include "xyMatrix.h"



int maxLen,minLen;
int missCount;



void error(int id)
{
    if(id==1)
        printf("ERROR : Memory can't be allocated!!!\n\n");
    else if ( id == 2 )
        printf("ERROR : File not Found!!!\n\n");
    else if ( id == 3 )
        printf("ERROR : Can't create Output File!!!\n\n");
    else if ( id == 4 )
        printf("ERROR : Invalid Number of Arguments!!!\n\n");
    system("PAUSE");
    exit(1);

    }
     

int recursiveClassify(timeSeries * T , ifstream &inFile , int nodeId )
{
    int i;	//search iterators
	int nextNodeId = -1;

    int shapeletLength;
    double shapeletDist;
    inFile >> shapeletLength;
    //cout << shapeletLength << " ";

    if(shapeletLength == 0)
    {

    	int classId;
    	inFile >> classId;
    	return classId;
    }

    inFile >> shapeletDist;
   // cout << shapeletDist << " ";
    double * y = (double *)malloc(sizeof(double)*shapeletLength);

    for( i = 0 ; i < shapeletLength ; i++ )
    	inFile >> y[i];

    timeSeries * S = new timeSeries(shapeletLength , y , -1 , 1);

	//xyMatrix * XY = new xyMatrix(T->length,shapeletLength);
	xyMatrix * XY = new xyMatrix(T->length,shapeletLength);

    XY->computeXY(T->x,T->length,y,shapeletLength);

	double mux , muy , sgx , sgy , smxy;
	int len = shapeletLength;

	muy = S->mean(0,len);
	sgy = S->stdv(0,len);

	double bsf = INF , t;
	int bestU = -1;


	for( int u = 0 ; u < T->length-len+1 ; u++ )
	{
	   mux = T->mean(u,len);
	   sgx = T->stdv(u,len);


	   if( sgx == 0 )
		   sgx = 1;


	   smxy = XY->sumXY(u,0,len);
		t = smxy - len*mux*muy;
		t = t / (len*sgx*sgy);
		t = sqrt(2*len*(1-t));
		if ( t < bsf )
		{
		   bsf = t;
		   bestU = u;
		}
	}

	free(y);

	//cout << "bsf : " << bsf << " ";

	if( bsf < shapeletDist )
		nextNodeId = 2*nodeId;
	else
		nextNodeId = 2*nodeId+1;

	while( 1 )
	{
	   inFile >> nodeId;
	  // cout << nodeId << " ";

	   if( nextNodeId == nodeId )
		   break;
	   inFile >> shapeletLength;

	    if(shapeletLength == 0)
	    {

	    	int classId;
	    	inFile >> classId;
	    	continue;
	    }

	    inFile >> shapeletDist;
	    double * y = (double *)malloc(sizeof(double)*shapeletLength);
	    for( i = 0 ; i < shapeletLength ; i++ )
	    	inFile >> y[i];
	    free(y);
	}

	delete(XY);
	delete(S);

	return recursiveClassify(T,inFile,nextNodeId);
}


int main(int argc , char ** argv )
{
    int i , j ;
    double t1,t2;
    t1 = clock();
    missCount = 0;


	if( argc < 4 )
	{
		printf("ERROR!!! usage: mueen_shapelet.exe train_file N C maxln minln stepsize\n");
		exit(1);
	}
	int N = atoi(argv[3]);
    int C = atoi(argv[4]);


	ifstream ifs ( argv[1] , ifstream::in );

    if( !ifs.is_open() )
        error(2);
    

    int * clsHist = (int *) malloc( sizeof(int) * C );
	for( i = 0 ; i < C ; i++ )
		clsHist[i] = 0;

	maxLen = 0;
	minLen = 9999999 ;
	
	int nodeId = -1;

	int ** confusionMatrix = (int **)malloc(sizeof(int *)*C);
	for( i = 0 ; i < C ; i++ )
	{
		confusionMatrix[i] = (int *)malloc(sizeof(int)*C);
		for( j = 0 ; j < C ; j++ )
			confusionMatrix[i][j] = 0;
	}



	//Read the input file
    timeSeries ** T = (timeSeries **)malloc(sizeof(timeSeries *)*N);
    for( j = 0 ; j < N && !ifs.eof() ; j++ )
    {

		 
         char s[100000];
		 ifs.getline(s,100000);
		
		 stringstream ss (stringstream::in | stringstream::out);
		 ss << s;	  
		 double x[10000];
         int c;
		 double ccc;
		 ss >> ccc;
		 c = (int)ccc;
		 if( c < 0 )
			 c = 0;
		 c = c%C;
		 //c--;

		 for( i = 0 ; !ss.eof() ; i++ )
  			ss >> x[i];
  
		 //int n = (int)x[i-1];
		 int n = i;
		 if( n > maxLen ) maxLen = n;
		 if( n < minLen ) minLen = n;
         T[j] = new timeSeries(n , x , c, 1);
		 clsHist[c]++;
		 //cout << j << " "  << n << endl;

	    char fname[100];
	    strcpy(fname,argv[2]);
	    strcat(fname,"_tree");
	    ifstream inFile(fname, ios::in);
	    if(!inFile.is_open() || inFile.bad()) exit(1);
	    if(!inFile.eof())
	    	inFile >> nodeId;
	    else
	    	break;
	    //cout << endl << nodeId << " " << i << " ";
		c = recursiveClassify(T[j],inFile,nodeId);
		if( c != T[j]->c )
			missCount++;
		confusionMatrix[T[j]->c][c]++;

		inFile.close();

		delete T[j];

	}
	ifs.close();

    t2 = clock();
	#if !RESULT
		printf("maxlen %d minLen %d\n",maxLen,minLen);
	#endif


	for( i = 0 ; i < C ; i++ )
	{
		for( j = 0 ; j < C ; j++ )
			cout << confusionMatrix[i][j] << "\t";
	cout << endl;
	}
	cout << "Missed Count " << missCount << endl;
	cout << "Accuracy " << (double)(N-missCount)/N << endl;

	for( i = 0 ; i < C ; i++ )
			free(confusionMatrix[i]);

	free(confusionMatrix);
	free(clsHist);
	free(T);


    return 1;   
}
