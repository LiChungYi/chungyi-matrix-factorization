#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<assert.h>
#include<string.h>
#include<float.h>
#include<math.h>
#include<vector>

using namespace std;

class RatingList{
    public:
	
	class RatingTuple{
	    public:
		int userID, itemID;
		double rating, originalRating;
		RatingTuple(int a, int b, double c, double d):userID(a),itemID(b),rating(c), originalRating(d){}
	};

	vector<RatingTuple> ratingList;
	int maxUidP1, maxIidP1;
	double maxValue, minValue;
	
	RatingList(const char* fileName){
		maxUidP1= 1, maxIidP1 = 1;
		FILE* fp = fopen(fileName, "r");
		int u, i;
		double r;
		assert(NULL != fp);
		maxValue = -1000000;	//just a small value
		minValue =  1000000;    //just a large value
		while(fscanf(fp, "%d %d %lf", &u,&i,&r) == 3){
			//start from 1, shift to 0
			assert(u >= 1 && i >= 1);
			u -= 1; i -= 1;

			RatingTuple x(u,i,r,r);
			ratingList.push_back(x);
			maxUidP1 = maxUidP1 >= u+1? maxUidP1: u+1;
			maxIidP1 = maxIidP1 >= i+1? maxIidP1: i+1;

			if(r > maxValue) maxValue = r;
			if(r < minValue) minValue = r;
		}
		assert(-1 == fscanf(fp, "%d", &u));
		fprintf(stderr, "%s: maxUidP1 %d, maxIidP1 %d\n", fileName, maxUidP1, maxIidP1);
		fclose(fp);
	}
};


class MatrixFactorization{
    public:
	/***********************************************************************************/
	static const int nFEATURE = 50;
	static const double learningRate = 0.004;//0.0025;//0.0000003;//biasLearningRate;
	static double featureReg;
	static const int MONITER_ITERATION_NUM = 20;
	/***********************************************************************************/

	RatingList training, validation, testing;
	double globalMean, globalVariance;
	double (*userFeature)[nFEATURE], (*itemFeature)[nFEATURE], (*userFeatureStep)[nFEATURE], (*itemFeatureStep)[nFEATURE];
	double (*userFeaturePast)[nFEATURE], (*itemFeaturePast)[nFEATURE];
	const char* trainingName, *validationName, *testingName;

	inline void allocateMemory(int maxUidP1, int maxIidP1){
		userFeature = new double[maxUidP1][nFEATURE];
		userFeatureStep = new double[maxUidP1][nFEATURE];
		userFeaturePast = new double[maxUidP1][nFEATURE];
		itemFeature = new double[maxIidP1][nFEATURE];
		itemFeatureStep = new double[maxIidP1][nFEATURE];
		itemFeaturePast = new double[maxIidP1][nFEATURE];
	}
	
	void getMeanVar_scaleTraining(RatingList& training, double& globalMean, double& globalVariance){
		double squareSum = 0;
		globalMean = 0;
		for(vector<RatingList::RatingTuple>::iterator it = training.ratingList.begin(); it != training.ratingList.end(); it++){
			globalMean += it->rating;
			squareSum += it->rating * it->rating;
		}
		globalMean /= (double)training.ratingList.size();
		globalVariance = sqrt(squareSum/(double)training.ratingList.size() - globalMean*globalMean);
			
		for(vector<RatingList::RatingTuple>::iterator it = training.ratingList.begin(); it != training.ratingList.end(); it++)
			it->rating = (it->rating - globalMean)/globalVariance;
	}


	void randomInitializeFeature(int maxUidP1, int maxIidP1){
	//	srand((unsigned int)time(NULL));
		for(int i = 0; i < maxUidP1; ++i){
			for(int j = 0; j < nFEATURE; ++j)
				userFeature[i][j] = ((double)rand()/RAND_MAX/10 - 0.05)/nFEATURE;
		}
		for(int i = 0; i < maxIidP1; ++i){
			for(int j = 0; j < nFEATURE; ++j)
				itemFeature[i][j] = ((double)rand()/RAND_MAX/10 - 0.05)/nFEATURE;
		}
	}


	MatrixFactorization(const char* trainFile, const char* validFile, const char* testFile):training(trainFile), validation(validFile), testing(testFile){
		{
			int i;
			for(i = (int)strlen(trainFile)-1; i >= 0;--i)
				if(trainFile[i] == '/')
					break;
			trainingName = &trainFile[i+1];

			for(i = (int)strlen(validFile)-1; i >= 0;--i)
				if(validFile[i] == '/')
					break;
			validationName = &validFile[i+1];

			for(i = (int)strlen(testFile)-1; i >= 0;--i)
				if(testFile[i] == '/')
					break;
			testingName = &testFile[i+1];
		}
		//print setting
		fprintf(stderr, "nFEATURE = %d\n", nFEATURE);
		fprintf(stderr, "learningRate = %lf\n", learningRate);
		fprintf(stderr, "featureReg = %lf\n", featureReg);

		//some data: last time frame => no data in training
		if(training.maxUidP1 < validation.maxUidP1)
			training.maxUidP1 = validation.maxUidP1;
		if(training.maxUidP1 < testing.maxUidP1)
			training.maxUidP1 = testing.maxUidP1;
		if(training.maxIidP1 < validation.maxIidP1)
			training.maxIidP1 = validation.maxIidP1;
		if(training.maxIidP1 < testing.maxIidP1)
			training.maxIidP1 = testing.maxIidP1;
	
		fprintf(stderr, "train: max%lf min%lf\n", training.maxValue, training.minValue);
		fprintf(stderr, "valid: max%lf min%lf\n", validation.maxValue, validation.minValue);
		fprintf(stderr, "test: max%lf min%lf\n", testing.maxValue, testing.minValue);

//		assert(training.maxValue == validation.maxValue && training.minValue == validation.minValue);
//		assert(training.maxValue == testing.maxValue && training.minValue == testing.minValue);
		assert(training.maxUidP1 >= validation.maxUidP1 && training.maxIidP1 >= validation.maxIidP1);
		assert(training.maxUidP1 >= testing.maxUidP1 && training.maxIidP1 >= testing.maxIidP1);
	
		allocateMemory(training.maxUidP1, training.maxIidP1);
		
		globalMean = 0, globalVariance = 1;
		getMeanVar_scaleTraining(training, globalMean, globalVariance);
		fprintf(stderr, "globalMean = %lf, globalVariance = %lf\n", globalMean, globalVariance);

		randomInitializeFeature(training.maxUidP1, training.maxIidP1);
	}

	void updateModel(const RatingList& training){
		for(int userID = 0; userID < training.maxUidP1; ++userID)
			for(int k = 0; k < nFEATURE; ++k)
				userFeatureStep[userID][k] = -featureReg*userFeature[userID][k];
		for(int itemID = 0; itemID < training.maxIidP1; ++itemID)
			for(int k = 0; k < nFEATURE; ++k)
				itemFeatureStep[itemID][k] = -featureReg*itemFeature[itemID][k];

		for(vector<RatingList::RatingTuple>::const_iterator it = training.ratingList.begin(); it != training.ratingList.end(); it++){
			int userID = it->userID, itemID = it->itemID;
			double rating = it->rating;
			double epsilon = -rating;
			for(int k = 0; k < nFEATURE; ++k)
				epsilon += userFeature[userID][k] * itemFeature[itemID][k];
			for(int k = 0; k < nFEATURE; ++k){
				double tmp1 = userFeature[userID][k], tmp2 = itemFeature[itemID][k];
				userFeatureStep[userID][k] -= epsilon * tmp2;
				itemFeatureStep[itemID][k] -= epsilon * tmp1;
			}
		}

		double preRMSE, preMAE;
		calculateTrainingError(training, preRMSE, preMAE);
		double learningRateNow = learningRate;

		for(int userID = 0; userID < training.maxUidP1; ++userID){
			for(int k = 0; k < nFEATURE; ++k)
				userFeaturePast[userID][k] = userFeature[userID][k];
		}
		for(int itemID = 0; itemID < training.maxIidP1; ++itemID){
			for(int k = 0; k < nFEATURE; ++k)
				itemFeaturePast[itemID][k] = itemFeature[itemID][k];
		}

		//line search
		while(1){
			for(int userID = 0; userID < training.maxUidP1; ++userID){
				for(int k = 0; k < nFEATURE; ++k){
					assert(!isnan((float)userFeatureStep[userID][k]));
					userFeature[userID][k] = userFeaturePast[userID][k] + learningRateNow * userFeatureStep[userID][k];
				}
			}
			for(int itemID = 0; itemID < training.maxIidP1; ++itemID)
				for(int k = 0; k < nFEATURE; ++k){
					assert(!isnan((float)itemFeatureStep[itemID][k]));
					itemFeature[itemID][k] = itemFeaturePast[itemID][k] + learningRateNow * itemFeatureStep[itemID][k];
				}

			double nowRMSE, nowMAE;
			calculateTrainingError(training, nowRMSE, nowMAE);
			if(nowRMSE < preRMSE)
				break;
			learningRateNow /= 2;
			printf("preRMSE %lf nowRMSE %lf, learningRateNow = %lf\n", preRMSE * globalVariance, nowRMSE * globalVariance, learningRateNow);
		}

	}
/*
	destructor
*/
	void calculateValidOrTestError(const RatingList& inList, double& theRMSE, double& theMAE, const double meanHere, const double varianceHere){
		theRMSE = 0;
		theMAE = 0;
		for(vector<RatingList::RatingTuple>::const_iterator it = inList.ratingList.begin(); it != inList.ratingList.end(); it++){
			int userID = it->userID, itemID = it->itemID;
			double epsilon = 0;
			for(int j = 0; j < nFEATURE; ++j)
				epsilon += userFeature[userID][j] * itemFeature[itemID][j];
	
			epsilon *= varianceHere;
			epsilon += meanHere - it->rating;

			//clipped penalty
/*			if(it->originalRating == inList.maxValue && epsilon > 0)
				epsilon = 0;
			if(it->originalRating == inList.minValue && epsilon < 0)
				epsilon = 0;*/
			
			theRMSE += epsilon*epsilon;
			theMAE += fabs(epsilon);
		}
		theRMSE = sqrt(theRMSE/(double)inList.ratingList.size());
		theMAE /= (double)inList.ratingList.size();
	}

	void calculateTrainingError(const RatingList& inList, double& theRMSE, double& theMAE){
		calculateValidOrTestError(inList, theRMSE, theMAE, 0, 1);
	}

	double getCorrelation(double a[], int aSize, double b[], int bSize){
		assert(aSize == bSize);
		double corr = 0;
		double aLen = 0, bLen = 0;
		for(int i = 0; i < aSize; ++i){
			corr += a[i] * b[i];
			aLen += a[i] * a[i];
			bLen += b[i] * b[i];
		}
		return corr/sqrt(aLen)/sqrt(bLen);
	}
	int startOptimizating(int fixedIter){	
		double trainingRMSE, validationRMSE, testingRMSE, trainingMAE, validationMAE, testingMAE, preValidationRMSE = DBL_MAX, bestValidationRMSE = DBL_MAX;
		int iter = 1, bestIter = -1;
		
		double (*userBestFeature)[nFEATURE], (*itemBestFeature)[nFEATURE];
		userBestFeature = new double[training.maxUidP1][nFEATURE];
		itemBestFeature = new double[training.maxIidP1][nFEATURE];

		while(1){
			//calculate all 
			calculateTrainingError(training, trainingRMSE, trainingMAE);
			calculateValidOrTestError(validation, validationRMSE, validationMAE, globalMean, globalVariance);
			calculateValidOrTestError(testing, testingRMSE, testingMAE, globalMean, globalVariance);
			fprintf(stderr, "currentIter = %d, bestIter = %d\n", iter, bestIter);
			fprintf(stderr, "trainRMSE = %lf(MAE %lf), valRMSE %lf(MAE %lf), testRMSE %lf(MAE %lf)\n", trainingRMSE*globalVariance, trainingMAE*globalVariance, validationRMSE, validationMAE, testingRMSE, testingMAE);
			
			//save the currently best model
			if(bestValidationRMSE > validationRMSE){
				bestValidationRMSE = validationRMSE;
				bestIter = iter;
				for(int i = 0; i < training.maxUidP1; ++i){
					for(int j = 0; j < nFEATURE; ++j)
						userBestFeature[i][j] = userFeature[i][j];
				}
				for(int i = 0; i < training.maxIidP1; ++i){
					for(int j = 0; j < nFEATURE; ++j)
						itemBestFeature[i][j] = itemFeature[i][j];
				}
			}

			//stop criterion
			if(fixedIter == 0){	//look at validation to stop
				if(preValidationRMSE < validationRMSE  && iter > (bestIter+MONITER_ITERATION_NUM))
					break;
			}
			else{
				if(iter == fixedIter)
					break;
			}

			preValidationRMSE = validationRMSE;
			
			updateModel(training);

			++iter;
		}
	
		/*
		//calculate the correlation using the currently best model
		for(int i = 0; i < training.maxIidP1; ++i){
			for(int j = 0; j < training.maxIidP1; ++j){
				printf("%lf ", getCorrelation(itemBestFeature[i], nFEATURE, itemBestFeature[j], nFEATURE));
			}
			printf("\n");
		}*/
		
		printf("end iter: %d, bestIter: %d\n", iter, bestIter);
		calculateTrainingError(training, trainingRMSE, trainingMAE);
		calculateValidOrTestError(validation, validationRMSE, validationMAE, globalMean, globalVariance);
		calculateValidOrTestError(testing, testingRMSE, testingMAE, globalMean, globalVariance);
		if(fixedIter != 0)
			printf("end Iter: trainRMSE = %lf(MAE %lf), valRMSE %lf(MAE %lf), testRMSE %lf(MAE %lf)\n", trainingRMSE*globalVariance, trainingMAE*globalVariance, validationRMSE, validationMAE, testingRMSE, testingMAE);
	
		if(fixedIter == 0){//not the retrain, in retrain we can't look at testing RMSE to decide our model
			//copy the best mode back
			for(int i = 0; i < training.maxUidP1; ++i){
				for(int j = 0; j < nFEATURE; ++j)
					userFeature[i][j] = userBestFeature[i][j];
			}
			for(int i = 0; i < training.maxIidP1; ++i){
				for(int j = 0; j < nFEATURE; ++j)
					itemFeature[i][j] = itemBestFeature[i][j];
			}
		

			for(int k = 0; k < nFEATURE; ++k){
				double userLen = 0, itemLen = 0;
//				for(int userID = 0; userID < training.maxUidP1; ++userID)
//					if(fabs(userFeature[userID][k]) > userLen)
//						userLen = fabs(userFeature[userID][k]);
//				for(int itemID = 0; itemID < training.maxIidP1; ++itemID)
//					if(fabs(itemFeature[itemID][k]) > itemLen)
//						itemLen = fabs(itemFeature[itemID][k]);
				for(int userID = 0; userID < training.maxUidP1; ++userID)
					userLen += userFeature[userID][k] * userFeature[userID][k];
				for(int itemID = 0; itemID < training.maxIidP1; ++itemID)
					itemLen += itemFeature[itemID][k] * itemFeature[itemID][k];
				printf("%lf\n", sqrt(userLen * itemLen));
			}

			calculateTrainingError(training, trainingRMSE, trainingMAE);
			calculateValidOrTestError(validation, validationRMSE, validationMAE, globalMean, globalVariance);
			calculateValidOrTestError(testing, testingRMSE, testingMAE, globalMean, globalVariance);
			printf("reg = %lf, bestIter: trainRMSE = %lf(MAE %lf), valRMSE %lf(MAE %lf), testRMSE %lf(MAE %lf)\n", featureReg, trainingRMSE*globalVariance, trainingMAE*globalVariance, validationRMSE, validationMAE, testingRMSE, testingMAE);
		}

		char modelName[1000];
		if(fixedIter == 0)
			sprintf(modelName, "%s_vali_model.txt", trainingName);
		else
			sprintf(modelName, "%s_retrain_model.txt", trainingName);
		FILE *model = fopen(modelName, "w");
		fprintf(model, "UserLatent\n");
		for(int i = 0; i < training.maxUidP1; ++i){
			for(int j = 0; j < nFEATURE; ++j)
				fprintf(model, " %lf", userFeature[i][j]);
			fprintf(model, "\n");
		}
		fprintf(model, "ItemLatent\n");
		for(int i = 0; i < training.maxIidP1; ++i){
			for(int j = 0; j < nFEATURE; ++j)
				fprintf(model, " %lf", itemFeature[i][j]);
			fprintf(model, "\n");
		}
		fclose(model);
		
/*
		char corrFileName[1000];
		if(fixedIter == 0)
			sprintf(corrFileName, "%s_vali_corr.txt", trainingName);
		else
			sprintf(corrFileName, "%s_retrain_corr.txt", trainingName);
		FILE *corr = fopen(corrFileName, "w");
		//calculate the correlation using the currently best model
		for(int i = 0; i < training.maxIidP1; ++i){
			for(int j = 0; j < training.maxIidP1; ++j){
				if(i < j)
					fprintf(corr,"%d,%d,%lf\n", i+1, j+1,getCorrelation(itemBestFeature[i], nFEATURE, itemBestFeature[j], nFEATURE));
			}
		}
		fclose(corr);*/
		
		char tmpName[100];
		sprintf(tmpName, "pred_%s", validationName);
		FILE *validPred = fopen(tmpName, "w");
		for(vector<RatingList::RatingTuple>::iterator it = validation.ratingList.begin(); it != validation.ratingList.end(); it++){
			double pred = 0;
			for(int k = 0; k < nFEATURE; ++k)
				pred += userFeature[it->userID][k]*itemFeature[it->itemID][k];
			pred = pred*globalVariance + globalMean;
			fprintf(validPred, "%lf\n", pred);
		}
		fclose(validPred);


		sprintf(tmpName, "pred_%s", testingName);
		FILE *testPred = fopen(tmpName, "w");
		for(vector<RatingList::RatingTuple>::iterator it = testing.ratingList.begin(); it != testing.ratingList.end(); it++){
			double pred = 0;
			for(int k = 0; k < nFEATURE; ++k)
				pred += userFeature[it->userID][k]*itemFeature[it->itemID][k];
			pred = pred*globalVariance + globalMean;
			fprintf(testPred, "%lf\n", pred);
		}
		fclose(testPred);
		
		return bestIter;

	}
	
};




double MatrixFactorization::featureReg;

int main(int argc, char* argv[]){
	srand((unsigned)time(NULL));
	assert(argc == 5);
	MatrixFactorization::featureReg = atof(argv[1]);

	const char *train = argv[2], *valid = argv[3], *test = argv[4];

	printf("valid To stop\n");
	MatrixFactorization model(train, valid, test);
	model.startOptimizating(0);
}
