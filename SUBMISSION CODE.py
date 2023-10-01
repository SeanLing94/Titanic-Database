{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0d062bb1-473b-44e9-af24-a02924063b46",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#IMPORTING THE LIBRARIES\n",
    "#Linear Algebra\n",
    "import numpy as np \n",
    "\n",
    "#Data Preprocessing\n",
    "import pandas as pd \n",
    "\n",
    "#Data Visualization\n",
    "import seaborn as sns\n",
    "from matplotlib import pyplot as plt\n",
    "from matplotlib import style"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "74b78658-4e23-40e2-b36f-e6a6742612e8",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Obtain the downloaded data from https://www.kaggle.com/c/titanic\n",
    "test_df = pd.read_csv(\"test.csv\")\n",
    "train_df = pd.read_csv(\"train.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c4f09781-5519-4adf-b95d-783cc6227927",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   PassengerId  Survived  Pclass  \\\n",
      "0            1         0       3   \n",
      "1            2         1       1   \n",
      "2            3         1       3   \n",
      "3            4         1       1   \n",
      "4            5         0       3   \n",
      "5            6         0       3   \n",
      "6            7         0       1   \n",
      "7            8         0       3   \n",
      "8            9         1       3   \n",
      "9           10         1       2   \n",
      "\n",
      "                                                Name     Sex   Age  SibSp  \\\n",
      "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
      "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
      "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
      "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
      "4                           Allen, Mr. William Henry    male  35.0      0   \n",
      "5                                   Moran, Mr. James    male   NaN      0   \n",
      "6                            McCarthy, Mr. Timothy J    male  54.0      0   \n",
      "7                     Palsson, Master. Gosta Leonard    male   2.0      3   \n",
      "8  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female  27.0      0   \n",
      "9                Nasser, Mrs. Nicholas (Adele Achem)  female  14.0      1   \n",
      "\n",
      "   Parch            Ticket     Fare Cabin Embarked  \n",
      "0      0         A/5 21171   7.2500   NaN        S  \n",
      "1      0          PC 17599  71.2833   C85        C  \n",
      "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
      "3      0            113803  53.1000  C123        S  \n",
      "4      0            373450   8.0500   NaN        S  \n",
      "5      0            330877   8.4583   NaN        Q  \n",
      "6      0             17463  51.8625   E46        S  \n",
      "7      1            349909  21.0750   NaN        S  \n",
      "8      2            347742  11.1333   NaN        S  \n",
      "9      0            237736  30.0708   NaN        C  \n"
     ]
    }
   ],
   "source": [
    "#Preview top 10 rows of the imported dataset\n",
    "print(train_df.head(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "86e00c80-3886-4ce7-b26e-aa6c3a867413",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "#DataFrame's Basic Information\n",
    "train_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "487ffd5d-0415-4733-97de-02d4acb59c04",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#DataFrame's Summary Statistics\n",
    "train_df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2a4d771c-dce1-4f86-ac55-426bd321bb33",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 600x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Features Correlational Estimation:\n",
    "#Pclass/Sex vs Survived\n",
    "plt.figure(figsize=(6, 4))\n",
    "sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_df)\n",
    "plt.title('Survival Rates by Pclass and Sex')\n",
    "plt.xlabel('Pclass')\n",
    "plt.ylabel('Survival Rate')\n",
    "plt.legend(title='Sex')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "aa88bc0a-00bf-4124-89d0-f9a325776758",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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cOBAw93d3UhKSrruWHnvtb3vSZ7rfcc6duxo1K1b94a1+vj4GKNGjbphH8BRHBaE/P39tXXrVu3YsUMzZ85Uly5d9NNPP2n8+PGqVauWTp06dctjP/roo/naBg4cqEuXLmnFihXWtsWLF8vT01O9e/e+7lhubm76+9//rk8//VQZGRmSru0RWbJkibp06SJ/f39r3z179qhz587y9/eXq6ur3N3d1bdvX+Xm5uqnn366pXW5ePGi3N3d5e7urjJlyuif//ynevXqpRkzZtj0y8nJ0UsvvaQaNWrIw8NDbm5u8vDw0M8//6wDBw7cdDm//PKLDh48aD0nJScnx/po3769UlNTdejQIUlSgwYNtG7dOo0bN05btmzRpUuXHFqnP5/30qhRI4WFhWnz5s3WtpEjR+rkyZNauXKlpGuHWuLi4tShQ4cCXWkpSV988YV27Nhh81i1alW+fu3bt7c5FB0RESFJ1j17f25PTk62ab///vtVp04dm7bevXsrMzNTu3fvtratXLlSjRs3lo+Pj9zc3OTu7q53333X7ufTsmVLlS5d2u567d69Ww8++KCCgoL07bffKiwszDrt888/l8Vi0d///nebzzA4OFh16tSxXnG3bds2Xb58+bqfRUE4sr0HBwerQYMGNm21a9e2OVS3efPm676XjnjwwQfl7u6uEiVKqGPHjgoODta6deuse6TWrVunFi1aWD/Pgirod6xBgwbau3evhg4dqvXr19u9srdBgwaKj4/X9OnTtX37dtNPHcBfE+EKVlFRUXruuee0cuVKHT9+XKNHj9bRo0dv66R2e1eD3X///apfv74WL14s6VpA+uCDD9SlSxf5+fndcLyBAwfq8uXLWr58uSRp/fr1Sk1N1YABA6x9kpOT1aRJEx07dkxz5861Bsd58+ZJksMBJI+3t7c1GKxZs0bNmzfXhx9+qJkzZ9r0i4mJ0QsvvKCuXbtqzZo1+v7777Vjxw7VqVOnQMvOO9dkzJgx1jCX9xg6dKgkWQPvG2+8oeeee06rVq1SixYt5Ofnp65du+rnn38u0DoFBwfbbfvjIZF69eqpSZMm1vfv888/19GjRx06RFSnTh1FRUXZPGrWrJmv358/fw8Pjxu2//l8r+utjyTrOn366afq2bOnypUrpw8++EDbtm3Tjh07rNvWn93oisaNGzfqxIkTGjx4sEqVKmUz7cSJEzIMQ0FBQfk+x+3bt1s/w7y6blT7jTi6vf/xHyF5PD09bfqdPn36luv5o/fff187duzQnj17dPz4ce3bt0+NGze2Tk9PT7+li1YK+h0bP368XnnlFW3fvl3t2rWTv7+/Hn74YZtbkKxYsUL9+vXTO++8o+joaPn5+alv375KS0tzuC4gz1/unCsUjLu7uyZPnqzXX39d//3vf63tnp6eysrKytf/eucnXO+8tgEDBmjo0KE6cOCADh8+nC8gXU+NGjXUoEEDLV68WE899ZQWL16skJAQtW7d2tpn1apVunjxoj799FObf/nf7v2kXFxcrCewS1KrVq0UGRmpqVOn6oknnlBoaKgk6YMPPlDfvn310ksv2cx/6tSpfH+A7cm7qm78+PHq3r273T5555QUL17cer7OiRMnrHuxOnXqpIMHD950Wfb+gKSlpaly5co2bSNGjNBjjz2m3bt361//+peqVq2qVq1a3XT8O+166yP9X6j44IMPFB4ebr24Io+97Vq6/jYsSWPHjtWvv/6qvn37Kicnx+Z8rTJlyshisWjr1q3y9PTMN29eW15d16v9ZnsHC2N79/f3v+F7WVARERE235k/CwgI0P/+9z+H6yvod8zNzU0xMTGKiYnRuXPn9MUXX2jChAlq06aNUlJSrCfrz5kzR3PmzFFycrJWr16tcePG6eTJk0pISHC4NkBizxUkpaam2m3P270eEhJibatYsaL1ar48mzZt0oULFxxa5uOPPy4vLy/Fx8crPj5e5cqVswlINzJgwAB9//33+uabb7RmzRr169fP5sTovD+Gf/yDZhiGFi5c6FCNN+Pp6al58+bp8uXLNid7WyyWfH9M//Of/+S7eWJenz/vWahWrZqqVKmivXv35tvbk/coUaJEvnqCgoLUv39/Pf744zp06JB+//33m67D0qVLbV5/9913+u233/LduDLvBrPPPPOMvvjiCw0dOvSuvCDkxx9/1N69e23ali1bphIlSlivJs27Cewf609LS7N7teDNuLi46K233tLIkSPVv39/xcXFWad17NhRhmHo2LFjdj/DWrVqSbp26MzLy+u6n8XNFMb23qJFi+u+l2Zq166dNm/ebD3MXVAF/Y79UalSpdSjRw8NGzZMZ86csXsD2woVKmj48OFq1aqVzWFkwFHsuYLatGmj8uXLq1OnTqpe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ABRJDoern376SUuXLtWyZct09OhRtWjRQjNnzlT37t1VogS/RQAAAP7aHA5X1atXV1RUlIYNG6a//e1vCg4OLoy6AAAAiiSHwlVubq4WLFigHj16WG8qCgAAcDPJ02rdsWVVmLTfof6xsbH69NNPdfDgQXl7e6tRo0aaNWuWqlWrdkvLdyhcubq6asSIEWrVqhXhqggwc0POOecvyfX/Pz9u6tiOfgkAADDTV199pWHDhql+/frKycnRxIkT1bp1ayUlJal4ccd/99fhw4K1atXS4cOHFR4e7vDCAAAA7jYJCQk2rxcvXqzAwEDt2rVLTZs2dXg8h2/FMGPGDI0ZM0aff/65UlNTlZmZafMAAAAoyjIyMiTplo/SObznqm3btpKkzp072/yAs2EYslgsys3NvaVCAAAAnM0wDMXExOihhx5SzZo1b2kMh8PV5s2bb2lBAAAAd7vhw4dr3759+uabb255DIfDVbNmzW55YQAAAHerp59+WqtXr9bXX3+t8uXL3/I4Doerr7/++obTb+XELwAAAGcxDENPP/20PvvsM23ZsuW2L9pzOFw1b948X9sfz73inCsAAFCUDBs2TMuWLdO///1vlShRQmlpaZIkX19feXt7Ozyew1cLnj171uZx8uRJJSQkqH79+tqwYYPDBQAAADhTXFycMjIy1Lx5c5UtW9b6WLFixS2N5/CeK19f33xtrVq1kqenp0aPHq1du3bdUiEAAODedTffMNowDFPHc3jP1fUEBATo0KFDZg0HAABQJDm852rfvn02rw3DUGpqqmbOnKk6deqYVhgAAEBR5HC4qlu3riwWS75daA8++KAWLVpkWmEAAABFkcPh6siRIzavXVxcFBAQIC8vL9OKAgAAKKocDldhYWGFUQcAAMA9ocAntH///fdat26dTdv777+v8PBwBQYG6h//+IeysrJMLxAAAKAoKXC4mjJlis3J7Pv379egQYP0yCOPaNy4cVqzZo1iY2MLpUgAAICiosDhKjExUQ8//LD19fLly9WwYUMtXLhQMTExeuONN/TRRx8VSpEAAABFRYHD1dmzZxUUFGR9/dVXX6lt27bW1/Xr11dKSoq51QEAABQxBQ5XQUFB1isFr1y5ot27dys6Oto6/fz583J3dze/QgAAgCKkwFcLtm3bVuPGjdOsWbO0atUqFStWTE2aNLFO37dv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",
      "text/plain": [
       "<Figure size 608.875x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Embarked/Pclass vs Survived\n",
    "EmbarkedvsSurvived = sns.catplot(\n",
    "    data=train_df,\n",
    "    x='Embarked',\n",
    "    y='Survived',\n",
    "    hue='Pclass',\n",
    "    kind='bar',\n",
    "    aspect=1.1\n",
    ")\n",
    "EmbarkedvsSurvived.set_axis_labels('Embarked', 'Survival Rate')\n",
    "EmbarkedvsSurvived.fig.suptitle('Survival Rates by Embarked and Pclass')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "92da15c6-f43f-4708-a254-819cd0ba7f8a",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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gEzHx8dHmzZt0vbt2/Xhhx/q6aef1qFDhzR48GA98sgj9/Wt+bPPPpukrWfPnvr333/tJp6YMWOGPD091alTpztuy83NTZ07d9a3335r+zY8ISFBc+bM0dNPPy0fHx/bsrt27VLr1q3l4+MjV1dXubu7q2vXrkpISNChQ4fu6bnExsbK3d1d7u7uKlCggF5++WV16NBB77//vt1y8fHx+uCDD1ShQgV5eHjIzc1NHh4e+uuvv3TgwAGHj/P333/rzz//1AsvvGDbXuJP8+bNFRkZaTty9eijj2rlypV66623tH79ev3777+pek6Jj5Godu3aKlasmNatW2dr69+/v86ePavFixdLujkkKjw8XC1atHA4o5oxxjY8r1GjRpKkoKAgNWjQQN98843djIQbNmxQxYoVVaFCBbttdOzY0e72li1bdOHCBXXr1s3utbFarWratKm2b99uN6wrpcLCwmz7193dXa1bt06yzO3v58TX6fYheI8++qjKly+foqGDd9KpUye7oarFihVT7dq17fbNnUyZMkXVqlWTl5eX3Nzc5O7urp9++inZ91+LFi3k6upqu514xPf48eOSpIMHD+rUqVN3rCclmjdvroiICC1ZskSDBg3Sww8/rKVLl6p169Z2RxSXLVumihUrqkqVKnb7tkmTJkmGa5YqVUpffvmlli5dqpYtW6pevXrJDiMG8OAiOAHItIKDg/Xmm29q8eLFOnXqlAYOHKhjx47d1wQRyQ2Zefjhh1WjRg3bULWEhATNnTtXTz/9tPLnz3/X7fXs2VPXrl3TggULJEmrV69WZGSk3TC9iIgI1atXTydPntTEiRNtofDTTz+VpFSHi0Q5cuTQ9u3btX37dv3www9q0KCB5s+frw8//NBuubCwML3zzjtq06aNfvjhB23dulXbt29X5cqVU/TYiedLDRo0yO6DvLu7u0JDQyXJFmYnTZqkN998U0uXLtUTTzyh/Pnzq02bNvrrr79S9JwKFiyYbNutQ/2qVq2qevXq2V6/ZcuW6dixY8kOobvdzz//rKNHj+q5555TTEyMoqOjFR0drfbt2+vq1at252KdP39e/v7+SbZxe1vi69OuXbskr89HH30kY4xtuFpyihYtqqioKF29etWu/bXXXrPt3zsN9bq9/W7n7wQEBKR4yGRyUrJvkjN+/Hi9/PLLqlmzpr755hv99ttv2r59u5o2bZrs++/WLxwkydPTU9L/+kni492pnpTKkSOH2rRpo7Fjx2rDhg36+++/VaFCBX366af6448/JN3ct3v37k2yX3Pnzi1jTJIvcVq0aCF/f39du3ZNYWFhdgEQwIOPWfUAPBDc3d01bNgw/ec//9F///tfW7unp6euX7+eZPk7fZi704xjPXr0UGhoqA4cOKAjR44kCT93UqFCBT366KOaMWOGevfurRkzZiggIECNGze2LbN06VLFxsbq22+/VbFixWzt9zvtt4uLi20yCElq1KiRqlevrhEjRuiFF15QYGCgpJvn33Tt2lUffPCB3frnzp1Tvnz5HD5OgQIFJEmDBw/WM888k+wyZcuWlSTlypVLI0aM0IgRI3TmzBnb0adWrVrpzz//dPhYp0+fTratVKlSdm39+vXTc889p99//12ffPKJypQpYzuCdDfTpk2TdPPD/Pjx45O9P/G8Mx8fH7tJNu5UY+LrM3nyZLtZDm+VXABL1KhRI61Zs0YrVqywm0o+MDDQtg89PDySXff293Ni6IiMjEwyA+SpU6dstUqSl5dXsn3n3LlzdsslutO+uT3o3G7u3Llq0KCBwsPD7dovX7581/XuJPHx7lTPvSpatKheeuklDRgwQH/88YcefvhhFShQQDly5ND06dOTXef216lPnz66fPmyHn74YfXr10/16tXTQw89dM81AchcOOIEINO50wxUicN6AgICbG3Fixe3zXqX6Oeff9aVK1dS9ZgdO3aUl5eXZs6cqZkzZ6pw4cJ24eduevTooa1bt+qXX37RDz/8oG7dutl905z44Tbxm3Pp5pCxL7/8MlU1OuLp6alPP/1U165d06hRo+we/9bHlm5eK+v2axfd/s1+orJly6p06dLas2ePgoODk/3JnTt3knr8/f3VvXt3dezYUQcPHkxyRCU58+bNs7u9ZcsWHT9+3G6WN0m2iyO/9tpr+vHHHxUaGupwGu6LFy9qyZIlqlOnjtatW5fk54UXXtD27dttwbx+/fr673//q/3799ttJ/HoYqI6deooX7582r9//x1fnzsFH+nm9cH8/f31xhtv3Pfsaw0bNpQk2zWqEm3fvl0HDhzQk08+aWtLru8cOnTINuzydvPnz7ebefD48ePasmVLkn1zu+Tef3v37r3na6KVLVtWhQoVumM9jly+fPmOfx9u/xvTsmVLHT58WD4+Psnu11uHhk6dOlVz587VJ598ou+//17R0dEp+vIFwIODI04AMp0mTZqoSJEiatWqlcqVKyer1ardu3dr3Lhx8vb2Vv/+/W3LdunSRe+8847effdd1a9fX/v379cnn3yivHnzpuox8+XLp7Zt22rmzJmKjo7WoEGDkkxLfCcdO3ZUWFiYOnbsqOvXryc5v6RRo0by8PBQx44d9cYbb+jatWsKDw/XxYsXU1VjStSvX1/NmzfXjBkz9NZbbykoKEgtW7bUzJkzVa5cOVWqVEk7d+7U2LFjkxyRKFmypHLkyKF58+apfPny8vb2VkBAgAICAvT555+rWbNmatKkibp3767ChQvrwoULOnDggH7//Xfb+UY1a9ZUy5YtValSJT300EM6cOCA5syZo1q1ailnzpwO69+xY4d69eql5557TidOnNDbb7+twoUL24YEJnJ1ddUrr7yiN998U7ly5Up2Wu3bzZs3T9euXVO/fv2S/bDv4+OjefPmadq0afrPf/6jAQMGaPr06WrWrJlGjhwpf39/ffXVV7YjZ4nvD29vb02ePFndunXThQsX1K5dO/n5+SkqKkp79uxRVFRUkqMtt8qXL5+WLl2qVq1aqXLlynr55Zf12GOPydvbW+fPn9fGjRt1+vTpFJ2/U7ZsWb300kuaPHmyXFxc1KxZMx07dkzvvPOOAgMDNXDgQNuyXbp0UefOnRUaGqpnn31Wx48f15gxY+Tr65vsts+ePau2bdvqxRdf1KVLlzRs2DB5eXlp8ODBd62pZcuWeu+99zRs2DDbLIUjR45UUFCQ4uPjHT6n27m4uOi9995Tr169bPVER0dr+PDhKRqqd/DgQTVp0kTPP/+86tevr0KFCunixYtavny5vvjiCzVo0MD2Wg8YMEDffPONHn/8cQ0cOFCVKlWS1WpVRESE1qxZo9dee001a9bUvn371K9fP3Xr1s0WlqZNm6Z27dppwoQJGjBgQKqfJ4BMyJkzUwBAchYuXGg6depkSpcubby9vY27u7spWrSo6dKli9m/f7/dstevXzdvvPGGCQwMNDly5DD169c3u3fvvuOserfPjnWrNWvW2GbZOnToUJL7b59V71adOnUykkydOnWSvf+HH34wlStXNl5eXqZw4cLm9ddfNytXrrSbnc2Y1M+ql5x9+/YZFxcX06NHD2OM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9OjRatmypaQHvB84M7UheSmZVeTPP/80xYoVMyEhIcaYm996bd261cTGxqZ3ebiF1Wq17a+WLVsme4TvQZgl5kFz66ySxhgTGRlpdzsiIsK4urqaVatWZWRZ2UriPhgyZIhtJrdPPvnEeHh4GBcXF7No0SK75ZD2bh1tsHr1anP9+nW7+3/44QeTL18+s3fv3owuLVuIj483xhgzc+ZM4+LiYvLnz2/7xt2Ym0c3AgICzOeff+6sErOF2//GWK1WEx8fb9q1a2fatm1r10+Q9hI/45w8edKsW7fONGvWzAwaNMhumYcfftgMHTrUGeWlOSaHyCT27dtnm4kncYxnfHy8bXa8ROb/DxCWLVtW48aN05w5c/Txxx+rfv36evPNN23XcEL6SUhIsLuduL/279+vhx9+WNLNWRD79u2rhISEB2PM7gMkISHBNnNb4u3EMeyJ/WPjxo2qUaOGatSo4ZQaszJjjN0+8PT01Pr16/X0009r6NChmjp1qtq3b693331XkZGRdvsKaSchIUHu7u6220899ZTtKHfi/41du3apUqVKqlChglNqzMrMLYN16tatqzZt2shqtSomJsZ2v5ubm4oUKaJLly45q8wsL7n/BxaLRa6uripUqJAiIyMlPRiTDjyIjDG2vzcBAQGqWrWqTp8+rdatW0uS7TOpn5+fLly44LQ60xL/0Zzs8uXLaty4sVq2bKmaNWuqZcuWWr58uSTJzc1NLi4u2rBhgwYOHJhkcoennnpKVatW1RtvvKHixYtr7dq1tuFjSHuJgcnV1VURERGSboamhIQEHTp0SLGxsapRo4Z++uknVahQQT/88INiY2P5g30PfvzxxyQBNZGrq6v++usvPfroo4qMjJSrq6utX1gsFq1evVqjRo1S/fr1lTdvXl7/NHTrh5Ljx49LujkpxzfffKO4uDht3bpVXbp00aRJk3T48GEtXrzYyRU/2FLTD2798Oji4qJVq1Zp3rx5atGihW0IN9LGrf3g5MmTKlmypAYPHqz8+fNr0KBB2r59u+Li4rRixQpFR0fr8ccfd3bJD7TU/j9I/CDftWtXbd26VXv37uULzHRwaz9I/EyUN29excbGatmyZZIkLy8vrVixQidOnFCzZs2cWW7acdKRLpibEzm0bdvWNGvWzOzevdv8+OOPpmHDhiY4ONh88803xhhjBgwYYLy9vc2bb75pN+Tr/PnzpnXr1sbV1dU2JAb3b/PmzaZ169bm4MGDd1xm4sSJdiddG2PMli1bTPny5U3nzp2Ni4uLGTlyZEaUm+Xs3r3bFC9e3FgsFrN27dpkl/nkk09Mnjx5zAsvvGC70GRsbKxZvHixef75503u3LmZFOU+pKYPrFmzxhw5csQsWbIkyTDhY8eOpXepWda99oMrV66Y+fPnm2eeecZ4e3ub0aNHZ2TZWUpq+sHSpUuNMcbs2bPHlC5d2hQqVMjUrFnT5MqVy4wbNy6jSs5y7rUfJNq2bZupVKmS2bBhQ0aUmyWlph8sXrzYWK1WM2XKFGOxWEytWrVMq1atTJ48eczHH3+cgVWnL4KTEx0/ftwULFjQLF++3Na2f/9+06tXL1OmTBkzfvx48/zzzyf7B2Pr1q2mb9++Sc7twL2bNWuWKViwoLFYLKZDhw5JzheIi4szISEhplChQmbx4sV2902dOtVYLBZTuXJls2fPnowsO8s4ePCgadWqlenevbtp1aqVKVeuXJL394EDB8yzzz5rF1qNufmBcdmyZeall14yBw4cyMiys5T76QNIG/fbD7799lvTtWtXs3///owsO0u5l36Q+MVmRESE2bRpk5kxY4bd+U5InfvpB7c6c+ZMepeaZd3P/4Nvv/3WDB061ISFhWW5/8kEJyc6ePCgKVeunPn222/t2jdu3GiaNm1qWrVq5aTKsp9jx46Z3r17m+HDh5vVq1cbNzc38+mnn9otY7Vazbx588ypU6eSrL9p0ybz2WefZVS5WdKhQ4fMqFGjzPbt2825c+dMYGCgCQkJSXLi750mQEk8URv35ujRo/fVB5A2Dhw4cF/9gBPh78/hw4fpB5nAwYMH76sfMCnT/YmMjDS9evWiHySD4OREMTExplSpUuaNN96w+2eXkJBg/vOf/5jq1avb5r5H+oqLizPLli0zx48fN8YYM2rUKJM7d27b0aM7/RFObOePdNo4f/687ffly5cbFxcX89VXXzmxouzjxo0bZtmyZbYhdintA7h/c+fONTNnzrTdPnfunO13+kHGuPW6iPfyvwD3b/PmzXZHlS5cuGD7nX6QMRYuXGimTZtmjDH0gzsgODnZZ599ZnLkyGG7SFuiY8eOGT8/P/Pjjz86qbKsLTY21uzevfuOUyXHx8ebxx9/3NStW9dcunQpg6vL+i5fvmwmT55s5s6dm+S9b8z//igPGDDA+Pn52Q15IaymjUuXLpm3337bDBs2zMydOzfJkQr6QMZYuXKlsVgspkmTJmbdunV299EP0l90dLSpW7euCQgIsAusiegH6e/SpUumUaNGpnz58ubnn39OchSJfpD+oqOjTYMGDYzFYjHdu3c3xvzvItvG0A9uxax6Tvbyyy+rZs2a6tu3r44dO2Zr9/HxkYeHh86cOeO84rKoVatWqUCBAurZs6eOHDmS7DKurq6aOnWq9uzZo48//tjWHhMTw+xU92nJkiUqVaqU5s+fr1GjRunxxx/XDz/8YLdM4gxIH3zwgQoVKqSBAwcqPj5ex44d06xZs3Tx4kVmSboPS5YsUcmSJfXbb79p+/bt6tmzp7744gtJ/5u2lz6QMS5duiRfX19dv35dCxYssE2fbLVa6QfpbMGCBSpcuLDy5cunAwcOyMfHx3Zf4sxs9IP0dfjwYdWrV085cuTQihUrVK9ePeXMmdN2v7llNmH6QfpYsGCBAgIC5Ofnp3feeUfr1q1TXFyc3Nzc6AfJcWZqw02nT582BQoUMO3atTPfffeduXLlivnkk09MqVKl7jqTCVJv/fr1pnr16iY0NNTkyZPHDBw40MTExNxx+S+//NK4u7ubn3/+2Xz11VemefPmyR4hQcrMmzfPlClTxkyaNMnExcWZhIQE06NHD1OlSpU7rrN9+3aTK1cu88ILL5hcuXKZZ555xvz7778ZWHXWMmPGDFO8eHETHh5u+5a2c+fOpl69enbLJd5HH0hf8+bNMx999JGZNWuWqVSpkhkzZozd/YlHAukHaWv8+PHGYrGYGTNm2NoiIyPN1atX7ZZLHJVAP0gf33zzjWnatKnt6Mavv/5qfv31V7she7eev0o/SFvPPPOMyZkzp5k6daoxxpjvv//eFC9e3GzcuDHZ5ekHDNXLNLZs2WJatWplPD09TXBwsMmdO7eZM2eOs8vKcn777Tfz9ttvm6tXr5qFCxcai8ViFi1adNeJBZ5++mljsViMxWIxI0aMyMBqs54333zTDBw40Ny4ccP2wXzjxo0mICDgjtNXX7p0yVSsWNFYLBamGU8DixYtMvPnz7ebIWnIkCFm7Nix5tChQ8muQx9IP1OnTjUvvPCCMcaYbt26mUaNGplVq1aZ999/3245+kHa+uyzz0yZMmXMsmXLjNVqNc8995x59NFHTWBgoBkwYIDZvXt3knXoB2mvU6dOplu3bsYYY5566ilTqVIl4+vra2rWrGnGjh2bZHn6QdoaNGiQOXr0qO12ZGSkyZEjh1m4cKExxiR7OkN27wcEp0wkOjrabN261SxbtizZsda4f/Hx8SY6Otp2+/nnnzclSpQw+/btS3b5n376yQQGBprg4GCm900Dly9fNn///bdd28qVK02xYsXsxlMnOn78uClbtqwpUqRIlpvS1Fni4uLsrncSGhpqLBaLqVmzpsmdO7fp1auX2bVrlzHm5lEn+kD6Gj9+vOnRo4cx5uZ04lWqVDF58uQx+fPnNxcuXDAJCQn0g3Rw7do1M2DAAOPv72/8/PxM165dzfTp0824ceNMyZIlTfv27e1GfNAP0sfw4cNNhw4dzMCBA03btm3N4cOHzW+//Wbeeecd4+/vb1asWGGMMfSDdGa1Wm1fILds2dK0adMm2eXoBwQnZFO3fkgPCAgwnTp1ShJWT548aerUqWO6du2a0eVleVar1XbEadq0aaZBgwbGGJMkPB08eNBuKA3S1k8//WQeeeQRs3btWhMVFWWWLFliHn30UTN06FBjtVrN8ePHzeOPP04fSAeJ7/+5c+eavn37GmOMGTFihHF1dTX+/v7mrbfesn3b++eff9IP0sHRo0dNu3btzMiRI83169dt++Trr782pUuXto36OHbsGP0gnWzatMmULFnSVKhQwSxatMjWfvz4cfPCCy+Y559/3vaB/tChQ/SDDNC/f39Tv359c+HCBbtJN/h/cJPFmOx2VhdwU3x8vNzc3LRhwwY98cQTCg8P14svvqi///5bf//9t5o3b67IyEgVKlTI2aVmSYmv/4ABA3T8+HEtWbJEknTx4kWdP39epUqVcnKFWV9CQoISEhLk4eFha6tXr54KFiyoxYsX68aNGzp//jx9IB2NGTNGy5YtU3x8vE6ePKnp06dr1apV2rx5s3r16qWePXs6u8Qsbffu3fL391ehQoXsJiIoUqSIunTpotGjR9MP0lmnTp20YMECzZ49W507d7a1h4SE6OTJk1q+fLlcXV2dWGH2YLVa5eLiojVr1qhZs2Y6cuSIihUrZusXcXFxOnfuXLbvB8yqh2wrccaY+vXr67XXXtPw4cM1YMAAVapUSZs2bZKkbP8HIj25ublJkvbs2aOaNWtKkubNm6fChQtr8+bNziwt23BxcbELTRcuXFDevHn1xBNPSJI8PDzoA+msdu3a2r17t4KCgrR582Y9+eSTeuutt2SM0dWrV51dXpZXpUoV23s8MTT99ddfypcvn2rXri2JfpDepk+frjx58mjBggU6fPiwrd3NzU1VqlQhNGUQF5ebkaBcuXIqU6aMvv32W0n/6xfu7u70A0luzi4AcKbEPwiDBg3SuHHjNHv2bH322Wd8y5tBrl27pri4OFWqVEk9evTQwoULNX78eHXr1s3ZpWULt07he+HCBfXv31///POPLTgh/VWvXl1btmxR2bJl5e7uLqvVKh8fH3333Xfy8/NzdnnZzvnz5zVy5EjlzZtXVapUcXY52YKXl5c2btyoTp06qWnTpurWrZt+//13bd68WUuXLnV2edlOQECAcuTIoYiICGeXkikxVA/Z3qZNm9SoUSPVqFFDS5cutbuWB9LXiRMnVKxYMUlS1apVtXz5chUsWNDJVWUv+/bt05w5c/T9998rf/78+u677+Tr6+vssoAMtWfPHs2YMUPLli2Tn58f/cAJDh48qAkTJigmJkZxcXGaPHmy/P39nV1WtpI4XO/ll1/WhQsXtHDhQmeXlOkQnJDt7dixQ7/99pteffVVZ5eS7VitVgUEBOitt97SgAEDnF1OtmS1WvX666+rXLlyevHFF51dDuAUVqtVoaGhqly5sl5++WVnl5OtJZ7/Cuc5f/48XyLfAcEJgFPFxcXJ3d3d2WVka4nfMgLZGf0AgCMEJwAAAABwgK9WAAAAAMABghMAAAAAOEBwAgAAAAAHCE4AAAAA4ADBCQAAAAAcIDgBAAAAgAMEJwAAAABwgOAEAMhwW7Zskaurq5o2bZqhj3vjxg2NHTtW1apVU65cuZQ3b15VrlxZQ4cO1alTpzK0FgDAg4UL4AIAMlyvXr3k7e2tqVOnav/+/SpatGi6P+b169fVuHFj7d27VyNGjFCdOnWUN29eHT58WEuXLlW+fPk0evToZNe9ceOGPDw80r1GAEDmxREnAECGio2N1aJFi/Tyyy+rZcuWmjlzZpJlvv/+e5UuXVo5cuTQE088oVmzZslisSg6Otq2zJYtW/T4448rR44cCgwMVL9+/RQbG3vHx/3Pf/6jX375RT///LP69eun6tWrq1SpUmrSpInCw8P1wQcf2JZt0KCBXn31VYWFhalAgQJq1KiRJGnDhg169NFH5enpqUKFCumtt95SfHy8bb3ixYtrwoQJdo9bpUoVDR8+3HbbYrEoPDxczZo1U44cORQUFKTFixen7kUEAGQ4ghMAIEMtXLhQZcuWVdmyZdW5c2fNmDFDtw5+OHbsmNq1a6c2bdpo9+7d6t27t95++227bezbt09NmjTRM888o71792rhwoX65Zdf9Oqrr97xcefPn69GjRqpatWqyd5vsVjsbs+aNUtubm7avHmzPv/8c508eVLNmzdXjRo1tGfPHoWHh2vatGkaNWpUql+Dd955R88++6z27Nmjzp07q2PHjjpw4ECqtwMAyDgEJwBAhpo2bZo6d+4sSWratKmuXLmin376yXb/lClTVLZsWY0dO1Zly5bV888/r+7du9ttY+zYserUqZMGDBig0qVLq3bt2po0aZJmz56ta9euJfu4hw4dUtmyZe3a2rZtK29vb3l7e6t27dp295UqVUpjxoxR2bJlVa5cOX322WcKDAzUJ598onLlyqlNmzYaMWKExo0bJ6vVmqrX4LnnnlOvXr1UpkwZvffeewoODtbkyZNTtQ0AQMYiOAEAMszBgwe1bds2Pf/885IkNzc3dejQQdOnT7dbpkaNGnbrPfroo3a3d+7cqZkzZ9pCj7e3t5o0aSKr1aqjR4/e8fFvP6r02Wefaffu3erZs6euXr1qd19wcLDd7QMHDqhWrVp226hTp46uXLmif/75JwXP/n9q1aqV5DZHnAAgc3NzdgEAgOxj2rRpio+PV+HChW1txhi5u7vr4sWLeuihh2SMSRJwbp/HyGq1qnfv3urXr1+Sx7jTRBOlS5fWn3/+addWqFAhSVL+/PmTLJ8rV64kNdyprsR2FxeXJLXGxcUlW8/tbt82ACBz4YgTACBDxMfHa/bs2Ro3bpx2795t+9mzZ4+KFSumefPmSZLKlSun7du32627Y8cOu9vVqlXTH3/8oVKlSiX5udPsdx07dtTatWu1a9eue6q/QoUK2rJli10w2rJli3Lnzm0Lgr6+voqMjLTdHxMTk+wRsN9++y3J7XLlyt1TXQCAjEFwAgBkiGXLlunixYsKCQlRxYoV7X7atWunadOmSZJ69+6tP//8U2+++aYOHTqkRYsW2WbeSzwq8+abb+rXX3/VK6+8ot27d+uvv/7S999/r759+97x8QcOHKhatWqpYcOGmjhxon7//XcdPXpUq1ev1sqVK+Xq6nrX+kNDQ3XixAn17dtXf/75p7777jsNGzZMYWFhcnG5+e+0YcOGmjNnjjZt2qT//ve/6tatW7LbXbx4saZPn65Dhw5p2LBh2rZt210ntgAAOB/BCQCQIaZNm6annnpKefPmTXLfs88+q927d+v3339XUFCQvv76a3377beqVKmSwsPDbbPqeXp6SpIqVaqkDRs26K+//lK9evVUtWpVvfPOO7ahd8nx8vLSTz/9pLfeekszZsxQ3bp1Vb58eQ0YMEB16tTR0qVL71p/4cKFtWLFCm3btk2VK1dWnz59FBISoqFDh9qWGTx4sB5//HG1bNlSzZs3V5s2bVSyZMkk2xoxYoQWLFigSpUqadasWZo3b54qVKiQkpcRAOAkXAAXAJDpvf/++5oyZYpOnDjh7FLum8Vi0ZIlS9SmTRtnlwIASAUmhwAAZDqfffaZatSoIR8fH23evFljx45lKBsAwKkITgCATOevv/7SqFGjdOHCBRUtWlSvvfaaBg8e7OyyAADZGEP1AAAAAMABJocAAAAAAAcITgAAAADgAMEJAAAAABwgOAEAAACAAwQnAAAAAHCA4AQAAAAADhCcAAAAAMABghMAAAAAOEBwAgAAAAAH/g9Ot1ShKQBJbwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Define Age Groups for visualization\n",
    "age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]\n",
    "age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']\n",
    "\n",
    "#Categorize passengers into the Age Groups\n",
    "train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=age_bins, labels=age_labels, right=False)\n",
    "\n",
    "#Barplot for AgeGroup/Sex vs Survived\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x='AgeGroup', y='Survived', hue='Sex', data=train_df)\n",
    "plt.title('Survival Rates by Age Group and Sex')\n",
    "plt.xlabel('Age Group')\n",
    "plt.ylabel('Survival Rate')\n",
    "plt.legend(title='Sex')\n",
    "plt.xticks(rotation=35)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "019383b1-8b50-45fa-9e25-55d7d0d22c5d",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "SibSpParchSurvivalR = train_df.groupby(['SibSp', 'Parch'])['Survived'].mean().unstack()\n",
    "\n",
    "#Heatmap for SibSp/Parch vs Survived\n",
    "plt.figure(figsize=(8, 4))\n",
    "sns.heatmap(SibSpParchSurvivalR, annot=True, cmap='YlGnBu', fmt='.2f')\n",
    "plt.title('Survival Rates by SibSp and Parch')\n",
    "plt.xlabel('Number of Parents/Children (Parch)')\n",
    "plt.ylabel('Number of Siblings/Spouses (SibSp)')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "97ff2b2a-3a94-49a4-ae4b-4a59eda8aac8",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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r59CQkLUpUuXm64hr4KCgvTDDz/kaM/rBWW5sVgsevnllzVy5Ei7ecze3t45LmaTLv/x9tf3XvZ+cvP000/r008/VXx8vMqVK6fNmzdr6tSp162rRIkSat++vWbPnq1Ro0ZpxowZ8vb2tvsDuCDOMeAKCLeAC8jMzNSsWbNUuXJlffTRRznWL1++XOPHj9fKlSv14IMPqkSJEurQoYNmz56txo0bKykpKcdo34MPPqjRo0crKChIFStWvKG6sm9H5OHhYWu7cOGC5syZY9evadOmki5feV2vXj1b+2effZbjwq8HH3xQK1asUOXKle3CeH7IDghWq9Wu/f3338/zPh588EEtXLhQmZmZN3ULtnnz5qlPnz625fnz50uS3ZXyktSnTx+99957euSRR3TmzBn17t37ho/5V02aNFFgYKCmTZumLl265Pkm/Dd6HqtUqaLXX39dixcv1rZt23Ks9/DwUKNGjXT77bdr3rx52rZt21XDbX78LP+qefPm+uSTT7Rs2TK1a9fO1p79s7meo0eP5jrCeuTIEaWkpCgqKsrWVqFCBe3cudOu3969e7Vnz55cw+3VtGrVSmXKlNGMGTNUrly5HAH1Wp5++ml98sknWrFihebOnauOHTva/eFUEOcYcAWEW8AFrFy5UkeOHNHYsWNzhB9JqlmzpqZMmaLp06frwQcflHT5I+1Fixapd+/eKlu2rN0V99Ll2wEtXrxYTZs21csvv6zatWsrKytLhw4d0urVqzVgwIDrhre2bdtqwoQJiomJ0XPPPadTp07p7bffzvHLsEaNGnr88cc1fvx4eXh46N5779WuXbs0fvx4BQYGqkiR/82AGjlypOLj49WkSRP16dNHVatW1cWLF3XgwAGtWLFC06ZNc/hj9GxNmjRRiRIl9Pzzz2v48OHy9PTUvHnztGPHjjzvo0uXLpo3b57atGmjvn37qmHDhvL09NThw4f13//+V+3bt1fHjh2vuQ8vLy+NHz9eqampatCgge1uCa1bt9bdd99t17dKlSp64IEHtHLlSt19992qU6fODT333Pj5+Wn8+PF65plndN999+nZZ59VSEiI9u3bpx07dmjKlCm5bpfX87hz50717t1bjz76qCIjI+Xl5aVvvvlGO3fu1ODBgyVJ06ZN0zfffKO2bduqXLlyunjxou3ThL++Zm+kBkc89dRTmjhx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A+Th4SEPDw+lpaUpIiJC48aN02uvvVYQNQIAAAB54nC49fT0lMVikSSFhITo0KFDkqTAwEDbvwEAAABncHjObd26dbVlyxZVqVJFzZs317Bhw3Ty5EnNmTNHtWrVKogaAQAAgDxxeOR29OjRCgsLkyS9+eabCgoK0gsvvKDjx4/rgw8+yPcCAQAAgLxyaOTWGKPSpUurRo0akqTSpUtrxYoVBVIYAAAA4CiHRm6NMYqMjNThw4cLqh4AAADghjkUbosUKaLIyEidOnWqoOpBIcnyLKYsr/9/eBZzdjkAAAD5wuELysaNG6dXXnlFU6dOVc2aNQuiJhSC1KqtnV0CAABAvnM43D7xxBM6f/686tSpIy8vL/n4+Nit//PPP/OtOAAAAMARDofbSZMmFUAZAAAAwM1zONx27dq1IOoAAAAAbprD97mVpP379+v111/X448/ruPHj0uSVq1apV27duVrcQAAAIAjHA63a9euVa1atfT999/r888/V2pqqiRp586dGj58eL4XCAAAAOSVw+F28ODBGjVqlOLj4+Xl5WVrb968uTZt2pSvxQEAAACOcDjc/vjjj+rYsWOO9tKlS3P/WwAAADiVw+G2ePHiOnr0aI727du3q0yZMvlSFAAAAHAjHA63MTExevXVV5WUlCSLxaKsrCx9++23GjhwoJ566qmCqBEAAADIE4fD7VtvvaVy5cqpTJkySk1NVfXq1dW0aVM1adJEr7/+ekHUCAAAAOSJw/e59fT01Lx58zRy5Eht375dWVlZqlu3riIjIwuiPgAAACDPHA63a9euVXR0tCpXrqzKlSsXRE0AAADADXF4WkLLli1Vrlw5DR48WD/99FNB1AQAAADcEIfD7ZEjRzRo0CCtX79etWvXVu3atTVu3DgdPny4IOoDAAAA8szhcFuqVCn17t1b3377rfbv36/OnTtr9uzZqlChgu69996CqBEAAADIE4fD7ZUqVqyowYMHa8yYMapVq5bWrl2bX3UBAAAADrvhcPvtt9+qV69eCgsLU0xMjGrUqKHly5fnZ20AAACAQxy+W8Jrr72mBQsW6MiRI7rvvvs0adIkdejQQb6+vgVRHwAAAJBnDofbNWvWaODAgercubNKlSplty4hIUF33HFHftUGAAAAOMThcLtx40a75eTkZM2bN08fffSRduzYoczMzHwrDgAAAHDE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4bZz5846deqURo4cqaNHj6pmzZpasWKFypcv7+zScJOsVquGDx+eYxoJgGvjvQPcGN47tz6Lud79FAAAAIBbxC0/5xYAAADIRrgFAACA2yDcAgAAwG0QbgEAAOA2CLdwSevWrdNDDz2k8PBwWSwWLV261NklAS4vLi5ODRo0kL+/v4KDg9WhQwft2bPH2WUBLm/q1KmqXbu27YsbGjdurJUrVzq7LNwgwi1c0rlz51SnTh1NmTLF2aUAt4y1a9fqxRdf1Hfffaf4+HhdunRJrVq10rlz55xdGuDSypYtqzFjxmjLli3asmWL7r33XrVv3167du1ydmm4AdwKDC7PYrFoyZIl6tChg7NLAW4pJ06cUHBwsNauXaumTZs6uxzgllKyZEn985//VI8ePZxdChzkFl/iAADIKTk5WdLlX9IA8iYzM1Offvqpzp07p8aNGzu7HNwAwi0AuCFjjPr376+7775bNWvWdHY5gMv78ccf1bhxY128eFF+fn5asmSJqlev7uyycAMItwDghnr37q2dO3dqw4YNzi4FuCVUrVpVCQkJOnPmjBYvXqyuXbtq7dq1BNxbEOEWANzMSy+9pGXLlmndunUqW7ass8sBbgleXl667bbbJEn169fX5s2bNXnyZL3//vtOrgyOItwCgJswxuill17SkiVLtGbNGlWsWNHZJQG3LGOM0tLSnF0GbgDhFi4pNTVV+/btsy0nJiYqISFBJUuWVLly5ZxYGeC6XnzxRc2fP19ffPGF/P39lZSUJEkKDAyUj4+Pk6sDXNdrr72m1q1bKyIiQmfPntXChQu1Zs0arVq1ytml4QZwKzC4pDVr1qh58+Y52rt27aqZM2cWfkHALcBiseTaPmPGDHXr1q1wiwFuIT169NDXX3+to0ePKjAwULVr19arr76qli1bOrs03ADCLQAAANwG31AGAAAAt0G4BQAAgNsg3AIAAMBtEG4BAADgNgi3AAAAcBuEWwAAALgNwi0AAADcBuEWAAAAboNwCwBubMSIEbrjjjucWsOTTz6p0aNH59v+mjVrpn79+uXb/q4mLS1N5cqV09atWwv8WADyD+EWgMvo1q2bLBaLLBaLPD09ValSJQ0cOFDnzp1zdmkua/HixWrWrJkCAwPl5+en2rVra+TIkfrzzz+dXZokaefOnfryyy/10ksv2dqaNWtm+zlbrVZVqVJFo0ePVmZmphMrzclqtWrgwIF69dVXnV0KAAcQbgG4lAceeEBHjx7Vb7/9plGjRum9997TwIEDnV2W02RmZiorKyvXdUOHDlXnzp3VoEEDrVy5Uj/99JPGjx+vHTt2aM6cOYVcae6mTJmiRx99VP7+/nbtzz77rI4ePao9e/aoT58+ev311/X22287qcqri42N1fr16/Xzzz87uxQAeUS4BeBSrFarQkNDFRERoZiYGMXGxmrp0qWSpLlz56p+/fry9/dXaGioYmJidPz4cdu2p0+fVmxsrEqXLi0fHx9FRkZqxowZkqT09HT17t1bYWFh8vb2VoUKFRQXF2fbNjk5Wc8995yCg4MVEBCge++9Vzt27LCtz/54f86cOapQoYICAwPVpUsXnT171tbn7Nmzio2NVbFixRQWFqaJEyfm+Ag9PT1dgwYNUpkyZVSsWDE1atRIa9assa2fOXOmihcvruXLl6t69eqyWq06ePBgjvP0ww8/aPTo0Ro/frz++c9/qkmTJqpQoYJatmypxYsXq2vXrrme382bN6tly5YqVaqUAgMDFR0drW3bttn1GTFihMqVKyer1arw8HD16dPHtu69995TZGSkvL29FRISokceeeSqP8usrCx9+umnateuXY51vr6+Cg0NVYUKFdS7d2+1aNHC9nOWpG+//VbR0dHy9fVViRIldP/99+v06dO5HqcgXxdBQUFq0qSJFixYcNXnCcC1EG4BuDQfHx9lZGRIuhxE3nzzTe3YsUNLly5VYmKiunXrZuv7xhtvaPfu3Vq5cqV+/vlnTZ06VaVKlZIk/etf/9KyZcv0ySefaM+ePZo7d64qVKggSTLGqG3btkpKStKKFSu0detW1atXTy1atLD7eH///v1aunSpli9fruXLl2vt2rUaM2aMbX3//v317bffatmyZYqPj9f69etzBMenn35a3377rRYuXKidO3fq0Ucf1QMPPKBff/3V1uf8+fOKi4vTRx99pF27dik4ODjHeZk3b578/PzUq1evXM9b8eLFc20/e/asunbtqvXr1+u7775TZGSk2rRpYwvpn332mSZOnKj3339fv/76q5YuXapatWpJkrZs2aI+ffpo5MiR2rNnj1atWqWmTZvmehzp8pSEM2fOqH79+lftk+3Kn3NCQoJatGihGjVqaNOmTdqwYYMeeuihq05bKKjXRbaGDRtq/fr1130OAFyEAQAX0bVrV9O+fXvb8vfff2+CgoLMY489lmv/H374wUgyZ8+eNcYY89BDD5mnn346174vvfSSuffee01WVlaOdV9//bUJCAgwFy9etGuvXLmyef/9940xxgwfPtz4+vqalJQU2/pXXnnFNGrUyBhjTEpKivH09DSffvqpbf2ZM2eMr6+v6du3rzHGmH379hmLxWL++OMPu+O0aNHCDBkyxBhjzIwZM4wkk5CQkOvzyNa6dWtTu3bta/bJrrtOnTpXXX/p0iXj7+9v/v3vfxtjjBk/frypUqWKSU9Pz9F38eLFJiAgwO4cXMuSJUuMh4dHjnMeHR1tOyeZmZlm5cqVxsvLywwaNMgYY8zjjz9u7rrrrqvu98rtc5Nfr4tskydPNhUqVLjqegCuhZFbAC5l+fLl8vPzk7e3txo3bqymTZvqnXfekSRt375d7du3V/ny5eXv769mzZpJkg4dOiRJeuGFF7Rw4ULdcccdGjRokDZu3Gjbb7du3ZSQkKCqVauqT58+Wr16tW3d1q1blZqaqqCgIPn5+dkeiYmJ2r9/v61fhQoV7OaOhoWF2T7+/u2335SRkaGGDRva1gcGBqpq1aq25W3btskYoypVqtgdZ+3atXbH8fLyUu3ata95nowxslgseT6v2Y4fP67nn39eVapUUWBgoAIDA5Wammo7h48++qguXLigSpUq6dlnn9WSJUt06dIlSVLLli1Vvnx5VapUSU8++aTmzZun8+fPX/VYFy5ckNVqzbXO9957z/ZzbteunZ544gkNHz5c0v9GbvOqoF4X2Xx8fK75PAG4FsItAJfSvHlzJSQkaM+ePbp48aI+//xzBQcH69y5c2rVqpX8/Pw0d+5cbd68WUuWLJF0+WNpSWrdurUOHjyofv366ciRI2rRooXtYrR69eopMTFRb775pi5cuKDHHnvMNl80KytLYWFhSkhIsHvs2bNHr7zyiq02T09Pu1otFovtYi9jjK3tStnt2cfx8PDQ1q1b7Y7z888/a/LkybZ+Pj4+1w2uVapU0f79+20f5edVt27dtHXrVk2aNEkbN25UQkKCgoKCbOcwIiJCe/bs0bvvvisfHx/16tVLTZs2VUZGhvz9/bVt2zYtWLBAYWFhGjZsmOrUqaMzZ87keqxSpUrp/Pnztn1fKTY2VgkJCdq/f78uXLig6dOny9fX1/b886ogXxfZ/vzzT5UuXTrPNQFwMucOHAPA//x1WsKVtmzZYiSZQ4cO2drmzJljJJnt27fnus20adOMv79/rutWrVplJJlTp06Z1atXGw8PD5OYmHjV2nL7eH/ixImmfPnyxpj/TUv47LPPbOuTk5NNsWLFbB+h79mzx0gy69atu+pxZsyYYQIDA6+6Ptt3331nJJlJkybluv706dO51u3n52dmz55tWz506JCRZCZOnJjrfn755RcjyWzdujXHutTUVFO0aFGzePHiXLc9fvx4rj+f600r6NatW56nJRTk6yLbE088YZ544omr1gPAtRR1UqYGAIeUK1dOXl5eeuedd/T888/rp59+0ptvvmnXZ9iwYYqKilKNGjWUlpam5cuXq1q1apKkiRMnKiwsTHfccYeKFCmiTz/9VKGhoSpevLjuu+8+NW7cWB06dNDYsWNVtWpVHTlyRCtWrFCHDh3ydEGUv7+/unbtqldeeUUlS5ZUcHCwhg8friJFithGYatUqaLY2Fg99dRTGj9+vOrWrauTJ0/qm2++Ua1atdSmTZs8n49GjRpp0KBBGjBggP744w917NhR4eHh2rdvn6ZNm6a7775bffv2zbHdbbfdpjlz5qh+/fpKSUnRK6+8YjdSOnPmTGVmZqpRo0by9fXVnDlz5OPjo/Lly2v58uX67bff1LRpU5UoUUIrVqxQVlaW3dSLK5UuXVr16tXThg0bHPoiiSFDhqhWrVrq1auXnn/+eXl5eem///2vHn30UduFYNkK8nWRbf369Tn2CcB1MS0BwC2hdOnSmjlzpj799FNVr15dY8aMyXFfVC8vLw0ZMkS1a9dW06ZN5eHhoYULF0qS/Pz8NHbsWNWvX18NGjTQgQMHtGLFClv4XLFihZo2baru3burSpUq6tKliw4cOKCQkJA81zhhwgQ1btxYDz74oO677z7dddddqlatmry9vW19ZsyYoaeeekoDBgxQ1apV1a5dO33//feKiIhw+JyMHTtW8+fP1/fff6/7779fNWrUUP/+/VW7du2r3grs448/1unTp1W3bl09+eST6tOnj93dGIoXL64PP/xQd911l2rXrq2vv/5a//73vxUUFKTixYvr888/17333qtq1app2rRpWrBggWrUqHHVGp977jnNmzfPoedVpUoVrV69Wjt27FDDhg3VuHFjffHFFypaNOd4TEG+LiRp06ZNSk5OvuYtzwC4FosxV0wIAwDkm3PnzqlMmTIaP368evTo4exynOLixYuqWrWqFi5cqMaNGzu7HIc9+uijqlu3rl577TVnlwIgj5iWAAD5ZPv27frll1/UsGFDJScna+TIkZKk9u3bO7ky5/H29tbs2bN18uRJZ5fisLS0NNWpU0cvv/yys0sB4ABGbgEgn2zfvl3PPPOM9uzZIy8vL0VFRWnChAm2L0EAABQ8wi0AAADcBheUAQAAwG0QbgEAAOA2CLcAAABwG4RbAAAAuA3CLQAAANwG4RYAAABug3ALAAAAt0G4BQAAgNv4P32DDo9oTxDWAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 800x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#Barchart to show the Fare/Pclass vs Survived\n",
    "plt.figure(figsize=(8, 4))\n",
    "sns.barplot(data=train_df, x='Pclass', y='Fare', hue='Survived')\n",
    "plt.title('Average Fare by Pclass and Survival')\n",
    "plt.xlabel('Passenger Class (Pclass)')\n",
    "plt.ylabel('Average Fare')\n",
    "plt.legend(title='Survived')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "de73f707-2d04-45e7-bc03-6d045196ee95",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 13 columns):\n",
      " #   Column       Non-Null Count  Dtype   \n",
      "---  ------       --------------  -----   \n",
      " 0   PassengerId  891 non-null    int64   \n",
      " 1   Survived     891 non-null    int64   \n",
      " 2   Pclass       891 non-null    int64   \n",
      " 3   Name         891 non-null    object  \n",
      " 4   Sex          891 non-null    object  \n",
      " 5   Age          714 non-null    float64 \n",
      " 6   SibSp        891 non-null    int64   \n",
      " 7   Parch        891 non-null    int64   \n",
      " 8   Ticket       891 non-null    object  \n",
      " 9   Fare         891 non-null    float64 \n",
      " 10  Cabin        204 non-null    object  \n",
      " 11  Embarked     889 non-null    object  \n",
      " 12  AgeGroup     713 non-null    category\n",
      "dtypes: category(1), float64(2), int64(5), object(5)\n",
      "memory usage: 84.9+ KB\n"
     ]
    }
   ],
   "source": [
    "#Preview the dataset again\n",
    "train_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d8246af0-8e98-4eda-9db2-77880a0275e5",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0     7\n",
      "1    71\n",
      "2     7\n",
      "3    53\n",
      "4     8\n",
      "Name: Fare, dtype: int32\n"
     ]
    }
   ],
   "source": [
    "#Datatype Conversion : Fare\n",
    "\n",
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "    #Convert all missing value to \"0\"\n",
    "    dataset['Fare'] = dataset['Fare'].fillna(0)\n",
    "    #Convert float.64 to int.64\n",
    "    dataset['Fare'] = dataset['Fare'].astype(int)\n",
    "    \n",
    "#Previewing first 5 rows of Fare\n",
    "print(train_df['Fare'].head(5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "950dc323-550f-4131-9c43-35fc809f89fb",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0    1\n",
      "1    0\n",
      "2    0\n",
      "3    0\n",
      "4    1\n",
      "Name: Sex, dtype: int32\n"
     ]
    }
   ],
   "source": [
    "#Datatype Conversion : Sex\n",
    "#Required library for data conversion\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "#Create a LabelEncoder object\n",
    "label_encoder = LabelEncoder()\n",
    "\n",
    "#Fit and transform 'Sex' feature in both dataframes\n",
    "train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])\n",
    "test_df['Sex'] = label_encoder.transform(test_df['Sex'])\n",
    "\n",
    "#Previewing first 5 rows of Sex\n",
    "print(train_df['Sex'].head(5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "72c57da5-3ce7-4828-bceb-1b063f429618",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0    0.0\n",
      "1    1.0\n",
      "2    0.0\n",
      "3    0.0\n",
      "4    0.0\n",
      "5    2.0\n",
      "6    0.0\n",
      "7    0.0\n",
      "8    0.0\n",
      "9    1.0\n",
      "Name: Embarked, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "#Datatype Conversion : Embarked\n",
    "#Dictionary for Embarked\n",
    "ports = {\"S\": 0, \"C\": 1, \"Q\": 2}\n",
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "#Assigning Embarked to their corresponding variables\n",
    "    dataset['Embarked'] = dataset['Embarked'].map(ports)\n",
    "    \n",
    "#Previewing first 10 rows of Embarked\n",
    "print(train_df['Embarked'].head(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "b95afcbd-be61-45df-b7d8-6db9e7374753",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 13 columns):\n",
      " #   Column       Non-Null Count  Dtype   \n",
      "---  ------       --------------  -----   \n",
      " 0   PassengerId  891 non-null    int64   \n",
      " 1   Survived     891 non-null    int64   \n",
      " 2   Pclass       891 non-null    int64   \n",
      " 3   Name         891 non-null    object  \n",
      " 4   Sex          891 non-null    int32   \n",
      " 5   Age          714 non-null    float64 \n",
      " 6   SibSp        891 non-null    int64   \n",
      " 7   Parch        891 non-null    int64   \n",
      " 8   Ticket       891 non-null    object  \n",
      " 9   Fare         891 non-null    int32   \n",
      " 10  Cabin        204 non-null    object  \n",
      " 11  Embarked     889 non-null    float64 \n",
      " 12  AgeGroup     713 non-null    category\n",
      "dtypes: category(1), float64(2), int32(2), int64(5), object(3)\n",
      "memory usage: 77.9+ KB\n"
     ]
    }
   ],
   "source": [
    "#DataFrame after the DataType conversion\n",
    "train_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "cd57d15a-1bc1-4aca-ab80-d099eae45606",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Features Dropping:\n",
    "#Removal of Cabin from Train.csv and Test.csv\n",
    "train_df = train_df.drop(['Cabin'], axis=1)\n",
    "test_df = test_df.drop(['Cabin'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "4c1673c7-729b-450d-81d1-6a50e829467e",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Removal of PassengerId from Train.csv and Test.csv\n",
    "train_df = train_df.drop(['PassengerId'], axis=1)\n",
    "test_df = test_df.drop(['PassengerId'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "6eff78d7-ad9d-478b-8f3e-ae8441076ece",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Removal of Ticket from Train.csv and Test.csv\n",
    "train_df = train_df.drop(['Ticket'], axis=1)\n",
    "test_df = test_df.drop(['Ticket'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "a66e44b1-2a59-499b-8581-5765f56f9dcd",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 10 columns):\n",
      " #   Column    Non-Null Count  Dtype   \n",
      "---  ------    --------------  -----   \n",
      " 0   Survived  891 non-null    int64   \n",
      " 1   Pclass    891 non-null    int64   \n",
      " 2   Name      891 non-null    object  \n",
      " 3   Sex       891 non-null    int32   \n",
      " 4   Age       714 non-null    float64 \n",
      " 5   SibSp     891 non-null    int64   \n",
      " 6   Parch     891 non-null    int64   \n",
      " 7   Fare      891 non-null    int32   \n",
      " 8   Embarked  889 non-null    float64 \n",
      " 9   AgeGroup  713 non-null    category\n",
      "dtypes: category(1), float64(2), int32(2), int64(4), object(1)\n",
      "memory usage: 57.0+ KB\n"
     ]
    }
   ],
   "source": [
    "#DataFrame after the Features Dropping\n",
    "train_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "1687d4d0-b922-449d-b920-39cdf7356831",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Total NaN counts</th>\n",
       "      <th>Percentage %</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>AgeGroup</th>\n",
       "      <td>178</td>\n",
       "      <td>20.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>177</td>\n",
       "      <td>19.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>2</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Total NaN counts  Percentage %\n",
       "AgeGroup               178          20.0\n",
       "Age                    177          19.9\n",
       "Embarked                 2           0.2\n",
       "Survived                 0           0.0\n",
       "Pclass                   0           0.0"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Summary of NaN value in the DataSet\n",
    "\n",
    "#Total Counts of the missing value in the dataset\n",
    "total = train_df.isnull().sum().sort_values(ascending=False)\n",
    "percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100\n",
    "\n",
    "#Percentage of the missing value in the dataset in descending order\n",
    "percent_2 = (round(percent_1, 1)).sort_values(ascending=False)\n",
    "\n",
    "#Creating a table to Concatenate both created Variables\n",
    "missing_data = pd.concat([total, percent_2], axis=1, keys=['Total NaN counts', 'Percentage %'])\n",
    "\n",
    "#Preview of the top 5 missing value in descending order\n",
    "missing_data.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "398cb5d6-6f21-44e8-a3c3-0d9c6b04508c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of NaN values in 'Age' column: 0\n"
     ]
    }
   ],
   "source": [
    "#Data Imputation\n",
    "data = [train_df, test_df]\n",
    "\n",
    "#Mean & Standard Deviation Imputation for Age\n",
    "for dataset in data:\n",
    "    mean = train_df[\"Age\"].mean()\n",
    "    std = test_df[\"Age\"].std()\n",
    "    is_null = dataset[\"Age\"].isnull().sum()\n",
    "    #Assigning random integers to the NaN values in Age\n",
    "    rand_age = np.random.randint(mean - std, mean + std, size=is_null)\n",
    "    age_slice = dataset[\"Age\"].copy()\n",
    "    age_slice[np.isnan(age_slice)] = rand_age\n",
    "    dataset[\"Age\"] = age_slice\n",
    "    #Convert Age into Integers\n",
    "    dataset[\"Age\"] = train_df[\"Age\"].astype(int)\n",
    "    \n",
    "#Check if any remaining NaN value in 'Age' Feature\n",
    "print(\"Number of NaN values in 'Age' column:\", train_df['Age'].isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "01331494-0c98-45e4-83cd-70cacc02140f",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of NaN values in 'AgeGroup' column: 0\n"
     ]
    }
   ],
   "source": [
    "#Re-Categorize passengers into the Age Groups\n",
    "age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]\n",
    "age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80' , '81-90']\n",
    "\n",
    "#Re-assigning AgeGroup to all the new imputed Ages\n",
    "data = [train_df, test_df]\n",
    "for dataset in data:\n",
    "    dataset['AgeGroup'] = pd.cut(dataset['Age'], bins=age_bins, labels=age_labels,right=False)\n",
    "\n",
    "#Check if any remaining NaN value in 'AgeGroup' Feature\n",
    "print(\"Number of NaN values in 'AgeGroup' column:\", train_df['AgeGroup'].isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "b36e6601-66e1-423f-b230-3a61b7dbe7be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "21-30    289\n",
       "31-40    227\n",
       "11-20    133\n",
       "41-50    106\n",
       "0-10      62\n",
       "51-60     48\n",
       "61-70     19\n",
       "71-80      6\n",
       "81-90      1\n",
       "Name: AgeGroup, dtype: int64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Display the AgeGroup in category\n",
    "train_df['AgeGroup'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "9ed78557-7fa8-4c2b-bdcf-bbf8eb754ed3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2    289\n",
       "3    227\n",
       "1    133\n",
       "4    106\n",
       "0     62\n",
       "5     48\n",
       "6     19\n",
       "7      6\n",
       "8      1\n",
       "Name: AgeGroup, dtype: int64"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Converting AgeGroup to Integer to represent the attribute to the column\n",
    "data = [train_df, test_df]\n",
    "for dataset in data:\n",
    "    dataset['AgeGroup'] = dataset['AgeGroup'].cat.codes\n",
    "\n",
    "#Preview of new AgeGroup\n",
    "train_df['AgeGroup'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "05f65413-2719-42a3-b5bb-df690bd5cb75",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of NaN values in 'Embarked' column: 0\n"
     ]
    }
   ],
   "source": [
    "#Mode Imputation for Embarked\n",
    "#Statistical Summary for Embarked to determine the Mode\n",
    "train_df['Embarked'].describe()\n",
    "\n",
    "#Replacing '0' to the Missing Value\n",
    "common_value = 0 \n",
    "data = [train_df, test_df]\n",
    "for dataset in data:\n",
    "    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)\n",
    "\n",
    "for dataset in data:\n",
    "    #Final Conversion of DataType in Embarked to integers\n",
    "    dataset['Embarked'] = dataset['Embarked'].astype(int)\n",
    "\n",
    "#Check if any remaining NaN value in 'Embarked' Feature\n",
    "print(\"Number of NaN values in 'Embarked' column:\", train_df['Embarked'].isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "dc611b7e-523d-476f-b3fa-742fee854dbf",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 10 columns):\n",
      " #   Column    Non-Null Count  Dtype \n",
      "---  ------    --------------  ----- \n",
      " 0   Survived  891 non-null    int64 \n",
      " 1   Pclass    891 non-null    int64 \n",
      " 2   Name      891 non-null    object\n",
      " 3   Sex       891 non-null    int32 \n",
      " 4   Age       891 non-null    int32 \n",
      " 5   SibSp     891 non-null    int64 \n",
      " 6   Parch     891 non-null    int64 \n",
      " 7   Fare      891 non-null    int32 \n",
      " 8   Embarked  891 non-null    int32 \n",
      " 9   AgeGroup  891 non-null    int8  \n",
      "dtypes: int32(4), int64(4), int8(1), object(1)\n",
      "memory usage: 49.7+ KB\n"
     ]
    }
   ],
   "source": [
    "#DataFrame after the Data Imputation\n",
    "train_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "2ed1dc70-b45d-4aaa-a378-c8b43304ea0a",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    517\n",
       "2    185\n",
       "3    126\n",
       "4     40\n",
       "5     23\n",
       "Name: Title, dtype: int64"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Feature Engineering : Title\n",
    "#Dictionary for Name\n",
    "titles = {\"Mr\": 1, \"Miss\": 2, \"Mrs\": 3, \"Master\": 4, \"Rare\": 5}\n",
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "    #Extract the first name as title\n",
    "    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)\n",
    "    #Normalizing rare titles to 'Rare'\n",
    "    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\n",
    "                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')\n",
    "    #Replacing 'Mlle' with 'Miss'\n",
    "    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')\n",
    "    #Replacing 'Ms' with 'Miss'\n",
    "    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')\n",
    "    #Replacing 'Mme' with 'Mrs'\n",
    "    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')\n",
    "    #Converting Titles to Integers\n",
    "    dataset['Title'] = dataset['Title'].map(titles)\n",
    "    #Converting 'NaN' to '0' to show the distinction of empty value\n",
    "    dataset['Title'] = dataset['Title'].fillna(0)\n",
    "    \n",
    "#Preview of new Feature : Title\n",
    "train_df['Title'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "ace374ee-63f6-4922-8af7-bb1a4820c963",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Dropping Name in the features\n",
    "train_df = train_df.drop(['Name'], axis=1)\n",
    "test_df = test_df.drop(['Name'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "a4fdb3df-d097-4414-9f09-0c4e453fded1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2    289\n",
       "3    227\n",
       "1    133\n",
       "4    106\n",
       "0     62\n",
       "5     48\n",
       "6     19\n",
       "7      6\n",
       "8      1\n",
       "Name: AgeGroup, dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Feature Engineering : AgeGroup\n",
    "#Display the AgeGroup in category\n",
    "train_df['AgeGroup'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "efbdb7fe-9894-4436-8778-98d1775357eb",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    537\n",
       "0    354\n",
       "Name: Solo_Traveller, dtype: int64"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Feature Engineering : Solo_Traveller\n",
    "data = [train_df, test_df]\n",
    "for dataset in data:\n",
    "    #Combination of SibSp(No. of siblings/spouses) and Parch(No. of parents/children)\n",
    "    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']\n",
    "    #Not travel alone\n",
    "    dataset.loc[dataset['relatives'] > 0, 'Solo_Traveller'] = 0\n",
    "    #travel alone\n",
    "    dataset.loc[dataset['relatives'] == 0, 'Solo_Traveller'] = 1\n",
    "    #Final conversion of datatype to Integers\n",
    "    dataset['Solo_Traveller'] = dataset['Solo_Traveller'].astype(int)\n",
    "    \n",
    "#Preview of new Feature : Solo_Traveller\n",
    "train_df['Solo_Traveller'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "10c30f5a-5ea1-4771-85d7-ff33cd349b78",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(-0.001, 7.0]    241\n",
      "(7.0, 8.0]        70\n",
      "(8.0, 14.0]      146\n",
      "(14.0, 26.0]     165\n",
      "(26.0, 52.0]     123\n",
      "(52.0, 512.0]    146\n",
      "Name: FareCategories, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "#Perform qcut on the 'Fare' feature\n",
    "#New feature 'FareCategories' to categorize 'Fare'\n",
    "train_df['FareCategories'] = pd.qcut(train_df['Fare'], q=6)\n",
    "\n",
    "#Display the 'FareCategories' in an ascending order \n",
    "fare_counts = train_df['FareCategories'].value_counts().sort_index()\n",
    "print(fare_counts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "e6d3d9ef-06bc-4708-b2c9-02d04888fa5f",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    241\n",
       "3    165\n",
       "5    146\n",
       "4    123\n",
       "2    121\n",
       "1     95\n",
       "Name: Fare, dtype: int64"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = [train_df, test_df]\n",
    "\n",
    "#Categorizing the Fare according to FareCategories\n",
    "for dataset in data:\n",
    "    dataset.loc[ dataset['Fare'] <= 7, 'Fare'] = 0\n",
    "    dataset.loc[(dataset['Fare'] > 7) & (dataset['Fare'] <= 9), 'Fare'] = 1\n",
    "    dataset.loc[(dataset['Fare'] > 8) & (dataset['Fare'] <= 14), 'Fare']   = 2\n",
    "    dataset.loc[(dataset['Fare'] > 14) & (dataset['Fare'] <= 26), 'Fare']   = 3\n",
    "    dataset.loc[(dataset['Fare'] > 26) & (dataset['Fare'] <= 52), 'Fare']   = 4\n",
    "    dataset.loc[ dataset['Fare'] > 52, 'Fare'] = 5\n",
    "    dataset['Fare'] = dataset['Fare'].astype(int)\n",
    "    \n",
    "#Preview of new 'Fare'\n",
    "train_df['Fare'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "295a6dab-e4e5-443c-b7c6-de0e38ec0ef7",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Dropping the 'FareCategories'\n",
    "train_df = train_df.drop(['FareCategories'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "92901cd7-37e9-44e4-912e-c2f80a608fe0",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "data = [train_df, test_df]\n",
    "\n",
    "#Computing Individual Cost\n",
    "for dataset in data:\n",
    "    dataset['Individual_Cost'] = dataset['Fare']/(dataset['relatives']+1)\n",
    "    #Converting Individual cost to integers\n",
    "    dataset['Individual_Cost'] = dataset['Individual_Cost'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "6bb3b988-20b2-467c-a4e7-f0c8adf690c3",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 13 columns):\n",
      " #   Column           Non-Null Count  Dtype\n",
      "---  ------           --------------  -----\n",
      " 0   Survived         891 non-null    int64\n",
      " 1   Pclass           891 non-null    int64\n",
      " 2   Sex              891 non-null    int32\n",
      " 3   Age              891 non-null    int32\n",
      " 4   SibSp            891 non-null    int64\n",
      " 5   Parch            891 non-null    int64\n",
      " 6   Fare             891 non-null    int32\n",
      " 7   Embarked         891 non-null    int32\n",
      " 8   AgeGroup         891 non-null    int8 \n",
      " 9   Title            891 non-null    int64\n",
      " 10  relatives        891 non-null    int64\n",
      " 11  Solo_Traveller   891 non-null    int32\n",
      " 12  Individual_Cost  891 non-null    int32\n",
      "dtypes: int32(6), int64(6), int8(1)\n",
      "memory usage: 63.6 KB\n"
     ]
    }
   ],
   "source": [
    "#DataFrame after Feature Engineering\n",
    "train_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "69496e19-711b-4e73-86b1-00efffbd9224",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Dropping the Survived for X_train set\n",
    "X_train = train_df.drop(\"Survived\", axis=1)\n",
    "\n",
    "#Including the Survived for Y_train set\n",
    "Y_train = train_df[\"Survived\"]\n",
    "\n",
    "#Establishing the X_Test data\n",
    "X_test  = test_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "04c9aab5-c20b-4f4f-9d5d-efabe1eab1f8",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Machine Learning Modelling\n",
    "#K Nearest Neighbour:\n",
    "#Importing KNN Library\n",
    "from sklearn import linear_model\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "#K Parameter = 3\n",
    "KNN = KNeighborsClassifier(n_neighbors = 3) \n",
    "KNN.fit(X_train, Y_train)\n",
    "\n",
    "#Applying on test set\n",
    "Y_pred = KNN.predict(X_test)\n",
    "\n",
    "#Accuracy of KNN computed\n",
    "acc_KNN = round(KNN.score(X_train, Y_train) * 100, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "8b0e515e-4b40-4d3c-91f5-60a27f8b1928",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Machine Learning Modelling\n",
    "#Logistic Regression:\n",
    "#Importing LR Library\n",
    "from sklearn import linear_model\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "#Increase the number of iterations to 1000 as Warning on the first round\n",
    "LR = LogisticRegression(max_iter=1000)\n",
    "LR.fit(X_train, Y_train)\n",
    "\n",
    "#Applying on test set\n",
    "Y_pred = LR.predict(X_test)\n",
    "\n",
    "#Accuracy of LR computed\n",
    "acc_LR = round(LR.score(X_train, Y_train) * 100, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "b869268c-11a5-415f-94d9-2f48609b96bc",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Machine Learning Modelling\n",
    "#Decision Tree:\n",
    "#Importing DT Library\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "DT = DecisionTreeClassifier() \n",
    "DT.fit(X_train, Y_train)\n",
    "\n",
    "#Applying on test set\n",
    "Y_pred = DT.predict(X_test) \n",
    "\n",
    "#Accuracy of DT computed\n",
    "acc_DT = round(DT.score(X_train, Y_train) * 100, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "cecb30c6-2790-4a14-af12-6a1b6d13a9c5",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Machine Learning Modelling\n",
    "#Decision Tree:\n",
    "#Importing RF Library\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "#n=100 decision trees to be used in the RF ensemble\n",
    "RF = RandomForestClassifier(n_estimators=100)\n",
    "RF.fit(X_train, Y_train)\n",
    "\n",
    "#Applying on test set\n",
    "Y_prediction = RF.predict(X_test)\n",
    "\n",
    "#Accuracy of RF computed\n",
    "RF.score(X_train, Y_train)\n",
    "acc_RF = round(RF.score(X_train, Y_train) * 100, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "7828b54c-b5b1-4701-a77b-64419fada419",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Machine Learning Modelling\n",
    "#Stochastic Gradient Descent (SGD):\n",
    "#Importing SGD Library\n",
    "from sklearn import linear_model\n",
    "from sklearn.linear_model import SGDClassifier\n",
    "\n",
    "#maximum number of iterations=5 for optimization, algorithmn will\n",
    "#not stop when the consecutive iteration is smaller then 'to1'\n",
    "SGD = linear_model.SGDClassifier(max_iter=5, tol=None)\n",
    "SGD.fit(X_train, Y_train)\n",
    "Y_pred = SGD.predict(X_test)\n",
    "\n",
    "#Accuracy of SGD computed\n",
    "SGD.score(X_train, Y_train)\n",
    "acc_SGD = round(SGD.score(X_train, Y_train) * 100, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "5f0a8250-43d8-41e4-81a9-caa2bb6054ef",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Machine Learning Modelling\n",
    "#Gaussian Naive Bayes:\n",
    "#Importing GNB Library\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "GNB = GaussianNB() \n",
    "GNB.fit(X_train, Y_train)\n",
    "\n",
    "#Applying on test set\n",
    "Y_pred = GNB.predict(X_test)\n",
    "\n",
    "#Accuracy of GNB computed\n",
    "acc_GNB = round(GNB.score(X_train, Y_train) * 100, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "2d2ae098-2cdc-4cd1-b4f7-1daa31433f6e",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Machine Learning Modelling\n",
    "#Perceptron:\n",
    "#Importing PCT library\n",
    "from sklearn import linear_model\n",
    "from sklearn.linear_model import Perceptron\n",
    "\n",
    "#maximum number of iterations=100 for optimization\n",
    "PCT = Perceptron(max_iter=100)\n",
    "PCT.fit(X_train, Y_train)\n",
    "\n",
    "#Applying on test set\n",
    "Y_pred = PCT.predict(X_test)\n",
    "\n",
    "#Accuracy of PCT computed\n",
    "acc_PCT = round(PCT.score(X_train, Y_train) * 100, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "9ef6766f-2638-4ee9-a003-a650fa55e1aa",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                        Model  Score\n",
      "2               Decision Tree  95.85\n",
      "3               Random Forest  95.85\n",
      "0                         KNN  87.65\n",
      "6                  Perceptron  81.59\n",
      "1         Logistic Regression  81.26\n",
      "5                 Naive Bayes  78.45\n",
      "4  Stochastic Gradient Decent  68.01\n"
     ]
    }
   ],
   "source": [
    "#Model Evaluation:\n",
    "#Accuracy:\n",
    "results = pd.DataFrame({\n",
    "    \n",
    "    #Computing the accuracy score into their respective Models\n",
    "    'Model': ['KNN', 'Logistic Regression', \n",
    "              'Decision Tree','Random Forest', \n",
    "              'Stochastic Gradient Decent', \n",
    "              'Naive Bayes', 'Perceptron'],\n",
    "    \n",
    "    'Score': [acc_KNN, acc_LR, acc_DT, \n",
    "              acc_RF, acc_SGD, acc_GNB, \n",
    "              acc_PCT]})\n",
    "\n",
    "#Sorting the result by their accuracy score descendingly\n",
    "result_df = results.sort_values(by='Score', ascending=False)\n",
    "print(result_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "d36844d5-2bd1-4f4b-9a09-cf9096202842",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#Dropping the 'Solo_Traveller'\n",
    "train_df = train_df.drop(['Solo_Traveller'], axis=1)\n",
    "test_df = test_df.drop(['Solo_Traveller'], axis=1)\n",
    "\n",
    "#Dropping the 'Parch'\n",
    "train_df = train_df.drop(['Parch'], axis=1)\n",
    "test_df = test_df.drop(['Parch'], axis=1)\n",
    "\n",
    "#Dropping the 'SibSp'\n",
    "train_df = train_df.drop(['SibSp'], axis=1)\n",
    "test_df = test_df.drop(['SibSp'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "5bcce848-1ab7-4192-b76f-330fac6431b0",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 10 columns):\n",
      " #   Column           Non-Null Count  Dtype\n",
      "---  ------           --------------  -----\n",
      " 0   Survived         891 non-null    int64\n",
      " 1   Pclass           891 non-null    int64\n",
      " 2   Sex              891 non-null    int32\n",
      " 3   Age              891 non-null    int32\n",
      " 4   Fare             891 non-null    int32\n",
      " 5   Embarked         891 non-null    int32\n",
      " 6   AgeGroup         891 non-null    int8 \n",
      " 7   Title            891 non-null    int64\n",
      " 8   relatives        891 non-null    int64\n",
      " 9   Individual_Cost  891 non-null    int32\n",
      "dtypes: int32(5), int64(4), int8(1)\n",
      "memory usage: 46.2 KB\n"
     ]
    }
   ],
   "source": [
    "train_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "672637ce-f118-4f17-ba13-bb506e15b4c0",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "95.85 %\n",
      "oob score: 79.91 %\n"
     ]
    }
   ],
   "source": [
    "#Random Forest is re-trained to compare the Accuracy Score.\n",
    "RF = RandomForestClassifier(n_estimators=100, oob_score = True)\n",
    "RF.fit(X_train, Y_train)\n",
    "\n",
    "#Applying on test set\n",
    "Y_prediction = RF.predict(X_test)\n",
    "\n",
    "RF.score(X_train, Y_train)\n",
    "\n",
    "#Accuracy Score is computed again\n",
    "acc_RF = round(RF.score(X_train, Y_train) * 100, 2)\n",
    "print(round(acc_RF,2,), \"%\")\n",
    "\n",
    "#OOB Score computed\n",
    "print(\"oob score:\", round(RF.oob_score_, 4)*100, \"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "679fe9c6-ccd8-4e35-b172-a78bd0fe19b4",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 1000}\n"
     ]
    }
   ],
   "source": [
    "#Hyperparameters Tuning:\n",
    "#Parameteres provided\n",
    "param_grid = {\n",
    "    \"criterion\": [\"gini\", \"entropy\"],\n",
    "    \"min_samples_leaf\": [1, 5, 10, 25, 50, 70],\n",
    "    \"min_samples_split\": [2, 4, 10, 12, 16, 18, 25, 35],\n",
    "    \"n_estimators\": [100, 400, 700, 1000, 1500]\n",
    "}\n",
    "\n",
    "#Importing library for GridSearchCV\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "RF = RandomForestClassifier(\n",
    "    n_estimators=100, max_features='sqrt', oob_score=True, random_state=1, n_jobs=-1\n",
    ")\n",
    "#Estimating the best parameters for Random Forest\n",
    "nRF = GridSearchCV(estimator=RF, param_grid=param_grid, n_jobs=-1)\n",
    "nRF.fit(X_train, Y_train)\n",
    "\n",
    "#Best Hyperparameters for RF\n",
    "RFbest_params = nRF.best_params_\n",
    "print(RFbest_params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "b4a9c8e6-ed7a-4f4b-8d11-610d3ca470d0",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy score: 89.11 %\n",
      "oob score: 82.83 %\n"
     ]
    }
   ],
   "source": [
    "#Random Forest Classifier is fitting with New HyperParameters\n",
    "RF = RandomForestClassifier(criterion = \"entropy\", \n",
    "                                       min_samples_leaf = 1, \n",
    "                                       min_samples_split = 10,   \n",
    "                                       n_estimators=1000, \n",
    "                                       max_features='sqrt', \n",
    "                                       oob_score=True, \n",
    "                                       random_state=1, \n",
    "                                       n_jobs=-1)\n",
    "\n",
    "#Retrained the model\n",
    "RF.fit(X_train, Y_train)\n",
    "Y_prediction = RF.predict(X_test)\n",
    "\n",
    "RF.score(X_train, Y_train)\n",
    "\n",
    "#Accuracy Score is computed again\n",
    "acc_RF = round(RF.score(X_train, Y_train) * 100, 2)\n",
    "print(\"accuracy score:\", round(acc_RF,2,), \"%\")\n",
    "\n",
    "#OOB Score computed\n",
    "print(\"oob score:\", round(RF.oob_score_, 4)*100, \"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "ea15d0f9-c8f2-4198-9c28-dace4da2490b",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[486,  63],\n",
       "       [ 98, 244]], dtype=int64)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Importing Library for Confusion Matrix\n",
    "from sklearn.model_selection import cross_val_predict\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "predictions = cross_val_predict(RF, X_train, Y_train, cv=3)\n",
    "\n",
    "#Confusion Matrix Model Evaluation\n",
    "confusion_matrix(Y_train, predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "c67f1b2c-adc8-443f-b65d-6c95308ffa21",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.819304152637486\n"
     ]
    }
   ],
   "source": [
    "#Importing Library for Accuracy\n",
    "from sklearn.metrics import accuracy_score\n",
    "#Computing Accuracy score\n",
    "print(\"Accuracy:\", accuracy_score(Y_train, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "78b58034-d8f7-432b-9636-3a5d17072ac2",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision: 0.7947882736156352\n"
     ]
    }
   ],
   "source": [
    "#Importing Library for Precision\n",
    "from sklearn.metrics import precision_score\n",
    "#Computing Precision score\n",
    "print(\"Precision:\", precision_score(Y_train, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "84831662-df78-4dc7-8c84-ccffa100d130",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recall: 0.7134502923976608\n"
     ]
    }
   ],
   "source": [
    "#Importing Library for Recall\n",
    "from sklearn.metrics import recall_score\n",
    "#Computing Recall score\n",
    "print(\"Recall:\",recall_score(Y_train, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "f57b74b5-7b92-4c6e-874e-293585790ad4",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7519260400616332"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Importing Library for F1_SCore\n",
    "from sklearn.metrics import f1_score\n",
    "#Computing F1 score\n",
    "f1_score(Y_train, predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dbe0d701-b0e4-479d-81f8-0569bd205c93",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
