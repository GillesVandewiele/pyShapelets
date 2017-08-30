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

#define MAXREF 5
#define INF 9999999999999.99
#define DEBUG 1
#define RESULT 1


#include "timeSeries.h"
#include "xyMatrix.h"
#include "orderLine.h"

int start = 10 , end;	
int stepSize = 10;
int samplingRate = 1;
int maxLen,minLen;
char inFname[100];


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
     

double sumOfProds(double * x, int s1 , double * y, int s2 , int n)
{
	double sum = 0;

	for ( int i = 0 ; i < n ; i++ )
		sum += x[i+s1]*y[i+s2];
	return sum;
}



int recursiveShapelet(timeSeries ** T, int N , int C , int nodeId)
{
    int i , j , k , len;	//search iterators
    double t1,t2;			//timers
	long long count = 0, pruneCount = 0;   //performance counters

	long long bhulCount = 0;
	double t3, boundTime = 0;
	
	int refCount = 0;
    t1 = clock();
    


    int * clsHist = (int *) malloc( sizeof(int) * C );
	for( i = 0 ; i < C ; i++ )
		clsHist[i] = 0;
	for( i = 0 ; i < N ; i++ )
		clsHist[T[i]->c]++;

    xyMatrix ** XY = (xyMatrix **)malloc(sizeof(xyMatrix *)*N);
  
    double bsf = 0;

	orderLine * line = new orderLine(N,C,clsHist);
    orderLine * bestSplit = new orderLine();
    orderLine ** references = new orderLine * [MAXREF];

	 for( j = 0 ; j < MAXREF ; j++ )
		references[j] = new orderLine();

	 for( j = 0 ; j < N ; j++ )
		  XY[j] = new xyMatrix(maxLen,maxLen);


	//For all time series 
	for( k = 0 ; k < N ; k++ )
	{

		 //Generate the matrices self matrix is also included when j == k
		 for( j = 0 ; j < N ; j++ )

			 XY[j]->computeXY(T[k]->x,T[k]->length,T[j]->x,T[j]->length);


		 //For all possible lengths    
		 for(len = start ; len <= end ; len += stepSize )
		 {
				  refCount = 0;

				 //For all possible positions
				 for( i = 0; i < T[k]->length - len + 1 ; i++ )
				 {
					   count++;
					   t3 = clock();
					   //printf("Doing %dth time series at position %d with length %d\n",k,i,len);
					   //We have a candidate. So create an order line. then sort it. then try different split point and record it.

					   if( !T[k]->checkPos(i,len) )
						   continue;

					   double mux , muy , sgx , sgy , smxy;
					   mux = T[k]->mean(i,len);
					   sgx = T[k]->stdv(i,len);
					   if( sgx == 0 )
						   continue;

					   int yy = 0;
					   while( refCount >= MAXREF && yy < MAXREF )
					   {
							   int r = references[yy]->shapeletPos;
				   			   muy = T[k]->mean(r,len);
							   sgy = T[k]->stdv(r,len);
							   smxy = XY[k]->sumXY(i,r,len);
							   //smxy = sumOfProds(T[k]->x,i,T[k]->x,r,len);

							   if( sgy == 0 )
								   error(1);

								double t = smxy - len*mux*muy;
								t = t / (len*sgx*sgy);
								t = 2*len*(1-t);
								if( t <= 0 )
									t = 0;
								else
									t = sqrt(t);

								//double tt = references[yy]->UB(t,references[yy]->splitPos);
								double tt = references[yy]->TUB(t);
								//double tt = references[yy]->shiftEntropy(t);
								if( references[yy]->informationGain < bestSplit->informationGain )
										break;
								//else if( references[yy]->informationGain == bestSplit->informationGain  && bestSplit->gap > references[yy]->gap )  ///breaking ties
								//		    break;
								
								//printf("%lf ",references[yy]->informationGain);
								yy++;
					   }

					   t2 = clock();
					   boundTime += (t2-t3)/CLOCKS_PER_SEC;
					   if( refCount >= MAXREF && yy < MAXREF )
					   {
						   pruneCount++;
						   continue;
					   }


					   line->reset(k,i,len);
                       
					   for( j = 0 ; j < N ; j++ )
					   {

						   struct projection p;
						   p.cls = T[j]->c;
						   p.length = len;
						   p.tID = j;

                           
						   double bsf = INF , t;
						   int bestU = -1;
						   for( int u = 0 ; u < T[j]->length-len+1 ; u++ )
						   {

							  // if(!T[j]->checkPos(u,len))
								 //  continue;

							   muy = T[j]->mean(u,len);
							   sgy = T[j]->stdv(u,len);

							   if( sgy == 0 )
								   sgy = 1;

							   smxy = XY[j]->sumXY(i,u,len);
							   //smxy = sumOfProds(T[k]->x,i,T[j]->x,u,len);
								t = smxy - len*mux*muy;
								t = t / (len*sgx*sgy);
								t = 2*len*(1-t);
								if( t <= 0 )
									t = 0;
								else
									t = sqrt(t);

								if ( t < bsf )
								{
								   bsf = t;
								   bestU = u;
								}
						   } 
						   p.pos = bestU;
						   p.distance = bsf;///sqrt((double)len);
						   line->insert(j , p);
						   //double bE = line->bestGain();
						   //if ( bE < bestSplit->informationGain )
							 //  break;
                           
					   }
					   line->findBestSplit();
					 //  printf("p%lf\n", line->informationGain);

					   for( int op = 0 ; refCount >= MAXREF && op < MAXREF ; op++ )
						   if( references[op]->informationGain < line->informationGain )
						   {
							   bhulCount++;
							   //printf("%d %d %d\n",k,len,i);
							   break;
							   
						   }
					   //if( refCount < MAXREF - 1)
					   //refCount = 0;

						if( line->informationGain > bestSplit->informationGain )
								bestSplit->copy(line);
						else if( line->informationGain == bestSplit->informationGain  && bestSplit->gap < line->gap )  ///breaking ties
							    bestSplit->copy(line);

						//TRy if replacing the best one works better or not.
						double minimumgain = 0;
						int minref = refCount%MAXREF;
						for( int er = 0 ;refCount >= MAXREF && er < MAXREF ; er++ )
							if(references[er]->tIG > minimumgain)
							{
								minimumgain = references[er]->tIG;
								minref = er;
							}



					//	if( line->informationGain < minimumgain )
						if(line->informationGain < bestSplit->informationGain)
						{
							//printf("%d \n",minref);
							references[minref]->copy(line);
							references[minref]->tIG = line->informationGain;
							refCount = refCount+1;
						}
						//else if( refCount < MAXREF )
						//if(line->informationGain < bestSplit->informationGain)
						//{
						//	references[refCount]->copy(line);
						///	references[refCount]->tIG = line->informationGain;
						//	refCount = refCount+1;
						//}
					   //else
					  // {
						//   int qq = rand()%MAXREF;
					//	   references[qq]->copy(line);
					  // }
				  }
		   }
		   t2 = clock();
#if !RESULT
		   printf("\nTime for %d th time series : %lf seconds\n",k,(t2-t3)/CLOCKS_PER_SEC);
		   printf("\nBest information gain : %lf \n",bestSplit->informationGain);
#endif



	  }               
    
	t2 = clock();
  
#if RESULT
	printf("Shpelet ID : %d , Start Position : %d , Shapelet Length : %d\n\n", bestSplit->shapeletID , bestSplit->shapeletPos , bestSplit->shapeletLength );
	printf("Split informationGain : %lf , split position %d , split distance %lf and the separation gap %lf\n\n",bestSplit->informationGain,bestSplit->splitPos,bestSplit->splitDist,bestSplit->gap);
	printf("Total candidates : %lld , ",count);
	printf("Number of Pruned candidates : %lld\n\n", pruneCount );
    printf("Time for current shapelet : %lf seconds and Bound Computation Time %lf\n\n\n",(t2-t1)/CLOCKS_PER_SEC,boundTime);
	printf("Bhool Count %ld\n\n\n",bhulCount);
#endif

	ofstream fout(inFname, ios::app);

	fout << nodeId << endl;
	fout  << bestSplit->shapeletLength*samplingRate << endl << bestSplit->splitDist*sqrt((double)samplingRate) << endl;
	int sId = bestSplit->shapeletID, s = bestSplit->shapeletPos;
	for( i = 0 ; i < bestSplit->shapeletLength ; i++ )
		for( j = 0 ; j < T[sId]->samplingRate ; j++ )
			fout << T[sId]->original_x[(s+i)*T[sId]->samplingRate+j] << " ";
	fout << endl;



	int sP = bestSplit->splitPos+1;
	int cc;


	bestSplit->findEntropies();


	if( bestSplit->leftEntropy != 0)
	{
		timeSeries ** LL = (timeSeries **)malloc(sizeof(timeSeries *)*sP);
		for( i = 0 ; i < C ; i++)
			clsHist[i] = 0;

		for( i = 0 ; i < sP  ; i++ )
		{
			LL[i] = T[bestSplit->line[i].tID];
			clsHist[LL[i]->c] = 1;
			//printf("%d ",bestSplit->line[i].tID);
		}


		cc = 0;
		k = 0;
		for( j = 0 ; j < C ; j++ )
		{
			cc += clsHist[j];
			if(clsHist[j] == 1)
			{
				for( i = 0 ; i < sP ; i++ )
					if( LL[i]->c == j )
						LL[i]->c = k;
				k++;
			}
		}

		recursiveShapelet(LL,sP,cc,2*nodeId);
	}
	else
	{
		fout << 2*nodeId << endl;
		fout << "0" << endl << T[bestSplit->line[sP-1].tID]->originalClass << endl;
	}


	if( bestSplit->rightEntropy != 0 )
	{
		cc = 0;

		timeSeries ** RR = (timeSeries **)malloc(sizeof(timeSeries *)*(N-sP));

    	for( i = 0 ; i < C ; i++)
	    	clsHist[i] = 0;

		for( i = 0 ; i < (N-sP)  ; i++ )
		{
			RR[i] = T[bestSplit->line[i+sP].tID];
			clsHist[RR[i]->c] = 1;
			//printf("%d ",bestSplit->line[i+sP].tID);
		}

		cc = 0;
		k = 0;
		for( j = 0 ; j < C ; j++ )
		{
			cc += clsHist[j];
			if(clsHist[j] == 1)
			{
				for( i = 0 ; i < (N-sP) ; i++ )
					if( RR[i]->c == j )
						RR[i]->c = k;
				k++;
			}
		}

		recursiveShapelet(RR,N-sP,cc,2*nodeId+1);
	}
	else
	{
		fout << 2*nodeId+1 << endl;
		fout << "0" << endl << T[bestSplit->line[sP].tID]->originalClass << endl;
	}


	 for( j = 0 ; j < MAXREF ; j++ )
		delete references[j];

	for( j = 0 ; j < N ; j++ )
		  delete XY[j];

	 delete line;
	 free(clsHist);
	 free(XY);
	 delete bestSplit;

	fout.close();


	return 1;
}


int main(int argc , char ** argv )
{
    int i , j ;
    double t1,t2;
    t1 = clock();

	if( argc < 4 )
	{
		printf("ERROR!!! usage: mueen_shapelet.exe train_file N C maxln minln stepsize\n");
		exit(1);
	}

	int N = atoi(argv[2]);
    int C = atoi(argv[3]);
	ifstream ifs ( argv[1] , ifstream::in );

    if( !ifs.is_open() )
        error(2);
    

    strcpy(inFname,argv[1]);

    int * clsHist = (int *) malloc( sizeof(int) * C );
	for( i = 0 ; i < C ; i++ )
		clsHist[i] = 0;

	maxLen = 0;
	minLen = 999999999 ;
	
	//Read the input file
    timeSeries ** T = (timeSeries **)malloc(sizeof(timeSeries *)*N);

    printf("Nr of classes: %d\n", C);

    for( j = 0 ; j < N && !ifs.eof() ; j++ )
    {
         char s[100000];
		 ifs.getline(s,100000);
		
		 stringstream ss (stringstream::in | stringstream::out);
		 ss << s;	  
		 double x[10000];
         double ccc;
		 ss >> ccc;
		 int c = (int)ccc;
		 if( c < 0 )
			 c = 0;
		 c = c%C;

		 printf("---->%d\n", c);

		 for( i = 0 ; !ss.eof() ; i++ )
  			ss >> x[i];
  
		 //int n = (int)x[i-1];
		 int n = i;
		 if( n > maxLen ) maxLen = n;
		 if( n < minLen ) minLen = n;
         T[j] = new timeSeries(n , x , c , samplingRate);
         T[j]->init();
		 clsHist[c]++;
    }
	ifs.close();

	t1 = clock();

	start = 10;
	end = maxLen;
	stepSize = 10;

	if( argc > 4 )	end = atoi(argv[4]);
	if( argc > 5 )	start = atoi(argv[5]);
	if( argc > 6 )	stepSize = atoi(argv[6]);


	#if RESULT
		printf("maxlen %d minLen %d\n",maxLen,minLen);
	#endif



	strcat(inFname,"_tree");
	ofstream fout(inFname);
	fout.close();
	recursiveShapelet(T,N,C,1);

	t2 = clock();
    printf("Total Execution Time : %lf\n\n",(t2-t1)/CLOCKS_PER_SEC);

	free(clsHist);
	for( i = 0 ; i < N ; i++ )
		delete T[i];
	free(T);


    return 1;   
}
