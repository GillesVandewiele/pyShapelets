
#include "timeSeries.h"
#include "xyMatrix.h"
#include "orderLine.h"
#include <stdio.h>
#include <string.h>

      
orderLine::orderLine()
{
     N = 0;
     nCls = 0;
     splitPos = -1;
	 splitDist = -1;
	 curN = -1;
	
	 entropy = 0;
     line = NULL;
     clsHist = leftClsHist = rightClsHist = NULL;
	 informationGain = 0;
	 gap = 0;
                       
};      
orderLine::orderLine(int n , int nC , int * cH)
{
     N = n;
     nCls = nC;
     splitPos = -1;
	 splitDist = -1;
     curN = 0;
	

     line = (projection *)malloc(sizeof(projection)*N);
     leftClsHist = (int *)malloc(sizeof(int)*nCls);
     rightClsHist = (int *)malloc(sizeof(int)*nCls);
     clsHist = (int *)malloc(sizeof(int)*nCls);
	 for( int i = 0 ; i < nCls ; i++ )
		 clsHist[i] = cH[i];

	 entropy = 0;	
	 double ratio;
     for( int i = 0 ; i < nCls ; i++ )
     {
          ratio = (double)(clsHist[i])/N;
          entropy += -(log(ratio)*ratio);
     }

	 printf("entropy is : %lf\n\n",entropy);

     };

void orderLine::copy(orderLine * src )
{
	      shapeletID = src->shapeletID;
	      shapeletPos = src->shapeletPos;
	      shapeletLength = src->shapeletLength;

	      N = src->N;
		  curN = src->curN;
	      splitPos = src->splitPos;
		  splitDist = src->splitDist;
	      nCls = src->nCls;
	      informationGain = src->informationGain;
	      gap = src->gap;


		  if( line == NULL )
			line = (projection *)malloc(sizeof(projection)*N);

	      for( int i = 0 ; i < N ; i++ )
	 		 line[i] = src->line[i];


		  if( leftClsHist == NULL )
		      leftClsHist = (int *)malloc(sizeof(int)*nCls);
		  if( rightClsHist == NULL )	
			  rightClsHist = (int *)malloc(sizeof(int)*nCls);
	      if( clsHist == NULL )
			  clsHist = (int *)malloc(sizeof(int)*nCls);

	      for( int i = 0 ; i < nCls ; i++ )
		  {
			  leftClsHist[i] = src->leftClsHist[i];
			  rightClsHist[i] = src->rightClsHist[i];
 	 		  clsHist[i] = src->clsHist[i];
		  }
	  	  entropy = 0;
	 	  double ratio;
	      for( int i = 0 ; i < nCls ; i++ )
	      {
	           ratio = (double)(clsHist[i])/N;
	           entropy += -(log(ratio)*ratio);
	      }
		  //printf("in copy\n");
};


void orderLine::reset(int sId, int sPos , int sLen)
{
	 shapeletID = sId;
     shapeletPos = sPos;
     shapeletLength = sLen;
     splitPos = -1;
	 splitDist = -1;
	 curN = 0;
	 rightTotal = leftTotal = 0;

	 projection p;
     p.cls = -1;
     p.length = 0;
     p.pos = -1;
     p.tID = -1;
     p.distance = 9999999.99;
	 

     for(int i = 0 ; i < N ; i++ )
           line[i] = p;

	 for( int i = 0 ; i < nCls ; i++ )
	 {
		rightClsHist[i] = 0;
	    leftClsHist[i] = 0;
	 }
	 //printf("LEft RESET\n");
	 informationGain = 0;
	 gap = 0;

};


void orderLine::insert(int i , projection p)
{

	if( curN == N )
	{
		printf("ERROR!!! line is full.\n");
	    exit(1);
	}

	//ordered insertion
	int j , k;
	for(  j = 0 ; j < curN ; j++ )
	{
		if( line[j].distance > p.distance )
		{
			for( k = curN-1 ; k >= j ; k-- )
				line[k+1] = line[k];
			line[j] = p;
			break;
		}
	}
	if( j == curN )
		line[j] = p;

	rightClsHist[p.cls]++;
	rightTotal++;

	if (curN == i) curN++;
	else printf("ERROR!!! insertion order missmatch\n");
};

double orderLine::minGap(int j)
{

	double meanLeft = 0 , meanRight = 0;
	for(int i = 0 ; i <= j ; i++)
		meanLeft += line[i].distance;

	meanLeft /= (j+1);

	for(int i = j+1 ; i < N ; i++)
		meanRight += line[i].distance;

	meanRight /= (N-j);

	return (meanRight-meanLeft)/sqrt((double)shapeletLength);
	/*
	if( j < curN )
	{
		//printf("%d %lf %lf\n",j , line[j+1].distance , line[j].distance );
		return line[j+1].distance - line[j].distance;
	}
	else
		return 0;
*/
};


double orderLine::gapDist(int j)
{
	if( j < curN )
	{
		//printf("%d %lf %lf\n",j , line[j+1].distance , line[j].distance );
		return (line[j+1].distance + line[j].distance)/2.0;
	}
	else
		return 0;

};


int comp(const void *a,const void *b)
{
    projection *x=(projection *)a;
    projection *y=(projection *)b;

    if (x->distance > y->distance )
        return 1;
    else if (x->distance < y->distance )
        return -1;
    else
        return 0;
    }


double orderLine::shiftEntropy(double shiftAmount)
{
	projection * tempLine;
	tempLine = (projection *)malloc(sizeof(projection)*N);
	memcpy(tempLine,line,sizeof(projection)*N);

	double maxInf = 0 , maxGap = 0 , maxDistance = 0;
	int maxi = -1;

	for( int q = 0 ; q < N ; q++ )
	{
	splitPos = q;

		for( int j = 1 ; j <= pow(2.0,nCls)-2 ; j++ )
		{
			//printf("Original Split Point %d ",splitPos);
			for(int i = 0 ; i < N ; i++ )
			{
				int k = j&(int)pow(2.0,line[i].cls);
				if( k == 0 )
					line[i].distance -= shiftAmount;
				else
					line[i].distance += shiftAmount;
			}
			qsort(line,N,sizeof(projection),comp);
			findEntropies();
			//findBestSplit();
			if( informationGain > maxInf )
			{
				maxi = splitPos;
				maxInf = informationGain;
				maxGap = gap;
				maxDistance = splitDist;
				//printf("splitPos : %d\n",splitPos);
			}
		   /* else if(informationGain == maxInf && gap > maxGap )
			{
				maxi = splitPos;
				maxInf = informationGain;
				maxGap = gap;
				maxDistance = splitDist;
				//printf("splitPos : %d\n",splitPos);
			}
	*/

			memcpy(line,tempLine,sizeof(projection)*N);

		}

	}
	splitPos = maxi;
	informationGain = maxInf;
	gap = maxGap;
	splitDist = maxDistance;
	free(tempLine);

	return maxInf;
};




double orderLine::TUB(double shiftAmount)
{
	projection * tempLine;
	tempLine = (projection *)malloc(sizeof(projection)*N);
	memcpy(tempLine,line,sizeof(projection)*N);

	double maxInf = 0 , maxGap = 0 , maxDistance = 0;
	int maxi = -1;

	int i;

	int *lch_tmp = (int*)malloc(sizeof(int)*nCls);
	int *rch_tmp = (int*)malloc(sizeof(int)*nCls);
	int ltot_tmp = 0;
	int rtot_tmp = N;

	for( i = 0; i < nCls ; i++ )
	{
	   leftClsHist[i] = rch_tmp[i] = lch_tmp[i] = 0;
	   rch_tmp[i] = rightClsHist[i] = clsHist[i];
	}
	leftTotal = 0;
	rightTotal = N;

	int * classType = (int *)malloc(sizeof(int)*nCls);
	

	int leftStart = 0, rightStart = 0;

   int k;
   for( k = 0 ; k < N-1 ; k++ )
   {
			int spPos = k;
            int c = line[k].cls;
            lch_tmp[c]++; leftClsHist[c]++;
			ltot_tmp++; leftTotal++;
            rch_tmp[c]--; rightClsHist[c]--;
			rtot_tmp--; rightTotal--;
	
			for(i = 0 ; i < nCls ; i++ )
			{
				if( leftClsHist[i] / leftTotal > rightClsHist[i] / rightTotal &&  (leftClsHist[i]+1) / (leftTotal+1) > (rightClsHist[i]-1) / (rightTotal-1) )
					classType[i] = -1;
				else if ( leftClsHist[i] / leftTotal < rightClsHist[i] / rightTotal &&  (leftClsHist[i]-1) / (leftTotal-1) < (rightClsHist[i]+1) / (rightTotal+1) )
					classType[i] = 1;
				else
					classType[i] = 0;
			}

			for( i = leftStart ; i <= spPos ; i++ )
				if( fabs(line[i].distance-line[spPos].distance) < shiftAmount )
					break;

			leftStart = i;

			for( i = rightStart ; i < N ; i++ )
				if( fabs(line[i].distance-line[spPos].distance) > shiftAmount )
					break;

			rightStart = i;
			

			for( i = leftStart ; i < rightStart ; i++ )
			{
				int c = line[i].cls;
				if( classType[c] == -1 && i > spPos )
				{
					leftClsHist[c]++;
					leftTotal++;
					rightClsHist[c]--;
					rightTotal--;
				}
				else if( classType[c] == 1 && i <= spPos )
				{
					leftClsHist[c]--;
					leftTotal--;
					rightClsHist[c]++;
					rightTotal++;
				
				
				}
			}
			informationGain = computeInformationGain();

			if( informationGain > maxInf )
			{
				maxInf = informationGain;
			}
			memcpy(line,tempLine,sizeof(projection)*N);
			memcpy(leftClsHist,lch_tmp,sizeof(int)*nCls);
			memcpy(rightClsHist,rch_tmp,sizeof(int)*nCls);
			leftTotal = ltot_tmp;
			rightTotal = rtot_tmp;

	}
	informationGain = maxInf;
	free(tempLine);
	free(classType);

	return maxInf;
};

double orderLine::UB(double shiftAmount, int spPos)
{
	projection * tempLine;
	tempLine = (projection *)malloc(sizeof(projection)*N);
	memcpy(tempLine,line,sizeof(projection)*N);

	double maxInf = 0 , maxGap = 0 , maxDistance = 0;
	int maxi = -1;

	int i;

	int *lch_tmp = (int*)malloc(sizeof(int)*nCls);
	int *rch_tmp = (int*)malloc(sizeof(int)*nCls);
	int ltot_tmp = 0;
	int rtot_tmp = N;

	for( i = 0; i < nCls ; i++ )
	{
	   leftClsHist[i] = rch_tmp[i] = lch_tmp[i] = 0;
	   rch_tmp[i] = rightClsHist[i] = clsHist[i];
	}
	leftTotal = 0;
	rightTotal = N;

	int * classType = (int *)malloc(sizeof(int)*nCls);
	

int k;
   for( k = 0 ; k < N-1 ; k++ )
   {
				spPos = k;

	//  for( k = 0 ; k <= spPos ; k++ )
	 // {
            int c = line[k].cls;
            lch_tmp[c]++; leftClsHist[c]++;
			ltot_tmp++; leftTotal++;
            rch_tmp[c]--; rightClsHist[c]--;
			rtot_tmp--; rightTotal--;
	//  }
	
			for(i = 0 ; i < nCls ; i++ )
			{
				if( leftClsHist[i] / leftTotal > rightClsHist[i] / rightTotal &&  (leftClsHist[i]+1) / (leftTotal+1) > (rightClsHist[i]-1) / (rightTotal-1) )
					classType[i] = -1;
				else if ( leftClsHist[i] / leftTotal < rightClsHist[i] / rightTotal &&  (leftClsHist[i]-1) / (leftTotal-1) < (rightClsHist[i]+1) / (rightTotal+1) )
					classType[i] = 1;
				else
					classType[i] = 0;
			}

			for( i = 0 ; i < N ; i++ )
			{
				int c = line[i].cls;
				if( classType[c] == -1 && i > spPos && (line[i].distance-line[spPos].distance) < shiftAmount )
				{
					leftClsHist[c]++;
					leftTotal++;
					rightClsHist[c]--;
					rightTotal--;
				}
				else if( classType[c] == 1 && i <= spPos && (line[spPos].distance-line[i].distance) < shiftAmount )
				{
					leftClsHist[c]--;
					leftTotal--;
					rightClsHist[c]++;
					rightTotal++;
				
				
				}
			}
/*
			for( i = 0 ; i < N ; i++ )
			{
				int c = line[i].cls;
				if( classType[c] == -1 && i > spPos )
					line[i].distance -= shiftAmount;
				else if( classType[c] == 1 && i <= spPos )
					line[i].distance += shiftAmount;
			}
			qsort(line,N,sizeof(projection),comp);
			splitPos = spPos;
			findEntropies();*/
/*
			for(i = 0 ; i < nCls ; i++ )
			{
				if( leftClsHist[i] / leftTotal > rightClsHist[i] / rightTotal &&  (leftClsHist[i]+1) / (leftTotal+1) > (rightClsHist[i]-1) / (rightTotal-1) && classType[i] == 1)
					return 999.00;
				else if ( leftClsHist[i] / leftTotal < rightClsHist[i] / rightTotal &&  (leftClsHist[i]-1) / (leftTotal-1) < (rightClsHist[i]+1) / (rightTotal+1) && classType[i] == -1)
					return 999.00;
			}

*/
			informationGain = computeInformationGain();

			if( informationGain > maxInf )
			{
				maxInf = informationGain;
			}
			memcpy(line,tempLine,sizeof(projection)*N);
			memcpy(leftClsHist,lch_tmp,sizeof(int)*nCls);
			memcpy(rightClsHist,rch_tmp,sizeof(int)*nCls);
			leftTotal = ltot_tmp;
			rightTotal = rtot_tmp;

	}
	informationGain = maxInf;
	free(tempLine);
	free(classType);

	return maxInf;
};

double orderLine::findEntropies()
{
       int i;
       for( i = 0; i < nCls ; i++ )
	   {
	      leftClsHist[i] = 0;
		  rightClsHist[i] = clsHist[i];
	   }
	   leftTotal = 0;
	   rightTotal = N;

       for( i = 0 ; i <= splitPos ; i++ )
       {
            int c = line[i].cls;
            leftClsHist[c]++;
			leftTotal++;
            rightClsHist[c]--;
			rightTotal--;
	   }
       informationGain = computeInformationGain();            

       return informationGain;
};
     


double orderLine::findBestSplit()
{
       int i;
       for( i = 0; i < nCls ; i++ )
	   {
	      leftClsHist[i] = 0;
		  rightClsHist[i] = clsHist[i];
	   }
	   leftTotal = 0;
	   rightTotal = N;

	   double maxInf = 0;
	   double maxGap = 0;
	   int maxi = -1;

       for( i = 0 ; i < N ; i++ )
       {
            int c = line[i].cls;
            leftClsHist[c]++;
			//printf("c : %d, lefthist[c] %d\n",c,leftClsHist[c]);
			leftTotal++;
            rightClsHist[c]--;
			rightTotal--;
            informationGain = computeInformationGain();            
			//printf("information Gain : %lf\n",informationGain);
            double mG = minGap(i);
            if( informationGain > maxInf )
			{ 
				maxi = i;
				maxInf = informationGain;
				maxGap = mG;
				//printf("splitPos : %d\n",splitPos);
			}
            else if(informationGain == maxInf && mG > maxGap )
			{
				maxi = i;
				maxInf = informationGain;
				maxGap = mG;
				//printf("splitPos : %d\n",splitPos);
			}

       }
      // printf("Shifted Split Point %d and informationGain %lf\n",maxi,maxInf);
       gap = maxGap;
       splitPos = maxi;
	   splitDist = gapDist(splitPos);
	   informationGain = maxInf;

       return maxInf;
};
     


double orderLine::computeInformationGain()
{
     double ratio;

     leftEntropy = 0;
     for( int i = 0 ; i < nCls ; i++ )
     {
          ratio = (double)leftClsHist[i]/leftTotal;
		  if(ratio>0)
			leftEntropy += -(log(ratio)*ratio);
	      //printf("%d\n", leftClsHist[i]);

     }

     rightEntropy = 0;
     for( int i = 0 ; i < nCls ; i++ )
     {
          ratio = (double)rightClsHist[i]/rightTotal;
		  if(ratio>0)
			rightEntropy += -(log(ratio)*ratio);
     }

	// printf("Left Entropy : %lf , Right Entropy : %lf , N %d \n",leftEntropy, rightEntropy,N);
     return entropy - (leftTotal/N)*leftEntropy - (rightTotal/N)*rightEntropy;
};

orderLine::~orderLine()
{
     if( line != NULL )
         free(line);
     if( leftClsHist != NULL )
         free(leftClsHist);
     if( rightClsHist != NULL )
         free(rightClsHist);

};
