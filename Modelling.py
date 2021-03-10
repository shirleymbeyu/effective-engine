{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Modelling.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyO3vhmx5jDHfO03Ok1iMg7v",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/shirleymbeyu/effective-engine/blob/main/Modelling.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Pf7fnyP2CGhJ"
      },
      "source": [
        " import pandas as pd\r\n",
        "import numpy as np\r\n",
        "import seaborn as sns\r\n",
        "import matplotlib.pyplot as plt "
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 456
        },
        "id": "xYMLo_mxCO7m",
        "outputId": "995976b4-9f60-447e-d0d3-dd076ee17948"
      },
      "source": [
        "df = pd.read_csv(\"credit_data (1).csv\")\r\n",
        "df"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Gender</th>\n",
              "      <th>Married</th>\n",
              "      <th>Dependents</th>\n",
              "      <th>Education</th>\n",
              "      <th>Self_Employed</th>\n",
              "      <th>ApplicantIncome</th>\n",
              "      <th>CoapplicantIncome</th>\n",
              "      <th>LoanAmount</th>\n",
              "      <th>Loan_Amount_Term</th>\n",
              "      <th>Credit_History</th>\n",
              "      <th>Property_Area</th>\n",
              "      <th>Default_Status</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Male</td>\n",
              "      <td>No</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>5849</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>360</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>Y</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>1</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>4583</td>\n",
              "      <td>1508</td>\n",
              "      <td>128</td>\n",
              "      <td>360</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Rural</td>\n",
              "      <td>N</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>Yes</td>\n",
              "      <td>3000</td>\n",
              "      <td>0</td>\n",
              "      <td>66</td>\n",
              "      <td>360</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>Y</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>0</td>\n",
              "      <td>Not Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>2583</td>\n",
              "      <td>2358</td>\n",
              "      <td>120</td>\n",
              "      <td>360</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>Y</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Male</td>\n",
              "      <td>No</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>6000</td>\n",
              "      <td>0</td>\n",
              "      <td>141</td>\n",
              "      <td>360</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>Y</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>606</th>\n",
              "      <td>Female</td>\n",
              "      <td>No</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>2900</td>\n",
              "      <td>0</td>\n",
              "      <td>71</td>\n",
              "      <td>360</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Rural</td>\n",
              "      <td>Y</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>607</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>3+</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>4106</td>\n",
              "      <td>0</td>\n",
              "      <td>40</td>\n",
              "      <td>180</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Rural</td>\n",
              "      <td>Y</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>608</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>1</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>8072</td>\n",
              "      <td>240</td>\n",
              "      <td>253</td>\n",
              "      <td>360</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>Y</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>609</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>2</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>7583</td>\n",
              "      <td>0</td>\n",
              "      <td>187</td>\n",
              "      <td>360</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>Y</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>610</th>\n",
              "      <td>Female</td>\n",
              "      <td>No</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>Yes</td>\n",
              "      <td>4583</td>\n",
              "      <td>0</td>\n",
              "      <td>133</td>\n",
              "      <td>360</td>\n",
              "      <td>0.0</td>\n",
              "      <td>Semiurban</td>\n",
              "      <td>N</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>611 rows × 12 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "     Gender Married Dependents  ... Credit_History Property_Area  Default_Status\n",
              "0      Male      No          0  ...            1.0         Urban               Y\n",
              "1      Male     Yes          1  ...            1.0         Rural               N\n",
              "2      Male     Yes          0  ...            1.0         Urban               Y\n",
              "3      Male     Yes          0  ...            1.0         Urban               Y\n",
              "4      Male      No          0  ...            1.0         Urban               Y\n",
              "..      ...     ...        ...  ...            ...           ...             ...\n",
              "606  Female      No          0  ...            1.0         Rural               Y\n",
              "607    Male     Yes         3+  ...            1.0         Rural               Y\n",
              "608    Male     Yes          1  ...            1.0         Urban               Y\n",
              "609    Male     Yes          2  ...            1.0         Urban               Y\n",
              "610  Female      No          0  ...            0.0     Semiurban               N\n",
              "\n",
              "[611 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7waK2SNzSvqX",
        "outputId": "b4cbf734-37cf-4d5d-a4a0-eef74abadf4c"
      },
      "source": [
        "df.info()"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 611 entries, 0 to 610\n",
            "Data columns (total 12 columns):\n",
            " #   Column             Non-Null Count  Dtype \n",
            "---  ------             --------------  ----- \n",
            " 0   Gender             611 non-null    object\n",
            " 1   Married            611 non-null    object\n",
            " 2   Dependents         611 non-null    object\n",
            " 3   Education          611 non-null    object\n",
            " 4   Self_Employed      611 non-null    object\n",
            " 5   ApplicantIncome    611 non-null    int64 \n",
            " 6   CoapplicantIncome  611 non-null    int64 \n",
            " 7   LoanAmount         611 non-null    int64 \n",
            " 8   Loan_Amount_Term   611 non-null    int64 \n",
            " 9   Credit_History     611 non-null    object\n",
            " 10  Property_Area      611 non-null    object\n",
            " 11  Default_Status     611 non-null    object\n",
            "dtypes: int64(4), object(8)\n",
            "memory usage: 57.4+ KB\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8fVkjWYcDjUu"
      },
      "source": [
        "## DATA PREPARATION"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "glZKeSJ8Bqfu"
      },
      "source": [
        "Our target varaible is default status and in order to work with it, we have to binarize it as: 0:N, 1:Y"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zfdYpGrL_wtD"
      },
      "source": [
        "# LabelBinarizer converts the string categorical variable to binary \r\n",
        "from sklearn.preprocessing import LabelBinarizer\r\n",
        "lb= LabelBinarizer()\r\n",
        "df[\"Default_Status\"]= lb.fit_transform(df[\"Default_Status\"])"
      ],
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 334
        },
        "id": "pRjyvUxDCDBu",
        "outputId": "e654414a-6b2c-4e43-e334-652b94551b58"
      },
      "source": [
        "# plotting risk distribution to understand whether there are more records \r\n",
        "# with more categories than the other.\r\n",
        "sns.countplot('Default_Status', data = df);"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/seaborn/_decorators.py:43: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
            "  FutureWarning\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wcLjCugWDsy7"
      },
      "source": [
        "Binning on the numeric variales: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YWd4wg4DDo0s"
      },
      "source": [
        "df['ApplicantIncome'] = pd.qcut(df.ApplicantIncome, q = 6)"
      ],
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "U_9p_OjZGpWO"
      },
      "source": [
        "interval = (0.0 , 10000, 20000, 30000, 41667)\r\n",
        "df['CoapplicantIncome'] = pd.cut(df.CoapplicantIncome, interval)"
      ],
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0oewkB9GHAMm"
      },
      "source": [
        "interval = (0, 140, 280, 320, 460, 700)\r\n",
        "df['LoanAmount'] = pd.cut(df.LoanAmount, interval)"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AbpWVXAPHPCj"
      },
      "source": [
        "interval = (0 ,96, 192, 288, 384, 480)\r\n",
        "df[\"Loan_Amount_Term\"] = pd.cut(df.Loan_Amount_Term, interval)"
      ],
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ML0Zp5u-OOPE",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 309
        },
        "outputId": "582c06c1-13dc-459a-ae84-8cafbfb3df50"
      },
      "source": [
        "df.head()"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Gender</th>\n",
              "      <th>Married</th>\n",
              "      <th>Dependents</th>\n",
              "      <th>Education</th>\n",
              "      <th>Self_Employed</th>\n",
              "      <th>ApplicantIncome</th>\n",
              "      <th>CoapplicantIncome</th>\n",
              "      <th>LoanAmount</th>\n",
              "      <th>Loan_Amount_Term</th>\n",
              "      <th>Credit_History</th>\n",
              "      <th>Property_Area</th>\n",
              "      <th>Default_Status</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Male</td>\n",
              "      <td>No</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>(4863.333, 6995.0]</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>(288, 384]</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>1</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>(3800.0, 4863.333]</td>\n",
              "      <td>(0.0, 10000.0]</td>\n",
              "      <td>(0.0, 140.0]</td>\n",
              "      <td>(288, 384]</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Rural</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>Yes</td>\n",
              "      <td>(2500.0, 3161.333]</td>\n",
              "      <td>NaN</td>\n",
              "      <td>(0.0, 140.0]</td>\n",
              "      <td>(288, 384]</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>0</td>\n",
              "      <td>Not Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>(2500.0, 3161.333]</td>\n",
              "      <td>(0.0, 10000.0]</td>\n",
              "      <td>(0.0, 140.0]</td>\n",
              "      <td>(288, 384]</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Male</td>\n",
              "      <td>No</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>(4863.333, 6995.0]</td>\n",
              "      <td>NaN</td>\n",
              "      <td>(140.0, 280.0]</td>\n",
              "      <td>(288, 384]</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  Gender Married Dependents  ... Credit_History Property_Area Default_Status\n",
              "0   Male      No          0  ...            1.0         Urban              1\n",
              "1   Male     Yes          1  ...            1.0         Rural              0\n",
              "2   Male     Yes          0  ...            1.0         Urban              1\n",
              "3   Male     Yes          0  ...            1.0         Urban              1\n",
              "4   Male      No          0  ...            1.0         Urban              1\n",
              "\n",
              "[5 rows x 12 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "u7-ysRx3P4OS"
      },
      "source": [
        "Tranforming our features into dummy variables by one-hot encode, hence making them robust for our linear regresssion model.\r\n",
        "we'll set the keyword drop_first to true so that one of the unique variables is deleted."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VfvCOaXmQKwV"
      },
      "source": [
        "#GENDER\r\n",
        "df = df.merge(pd.get_dummies(df.Gender, drop_first= True, prefix='sex'), left_index=True, right_index=True)"
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3V1J1_bWRDD-"
      },
      "source": [
        "#Married\r\n",
        "df = df.merge(pd.get_dummies(df.Married, drop_first= True, prefix='Married'), left_index=True, right_index=True)"
      ],
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bIOnlJ1oVAEY"
      },
      "source": [
        "#Dependents\r\n",
        "df = df.merge(pd.get_dummies(df.Dependents, drop_first= True, prefix='Dependents'), left_index=True, right_index=True)"
      ],
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KiCjckEuY3Ra"
      },
      "source": [
        "#Education\r\n",
        "df = df.merge(pd.get_dummies(df.Education, drop_first= True, prefix='Education'), left_index=True, right_index=True)"
      ],
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3GLwy95YVIyK"
      },
      "source": [
        "#Self_Employed\r\n",
        "df = df.merge(pd.get_dummies(df.Self_Employed, drop_first= True, prefix='Self_Employed'), left_index=True, right_index=True)"
      ],
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YH-WTWXeVRIY"
      },
      "source": [
        "#Credit_History\r\n",
        "df = df.merge(pd.get_dummies(df.Credit_History, drop_first= True, prefix='Credit_History'), left_index=True, right_index=True)"
      ],
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gaAuWZgEVV_7"
      },
      "source": [
        "#Property_Area\r\n",
        "df = df.merge(pd.get_dummies(df.Property_Area, drop_first= True, prefix='Property_Area'), left_index=True, right_index=True)"
      ],
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-riyrmLZWrtP"
      },
      "source": [
        "#ApplicantIncome\r\n",
        "df = df.merge(pd.get_dummies(df.ApplicantIncome, drop_first= True, prefix='ApplicantIncome'), left_index=True, right_index=True)"
      ],
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3i6PJI9VWwaY"
      },
      "source": [
        "#CoapplicantIncome\r\n",
        "df = df.merge(pd.get_dummies(df.CoapplicantIncome, drop_first= True, prefix='CoapplicantIncome'), left_index=True, right_index=True)"
      ],
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uJFsuM73Wx2E"
      },
      "source": [
        "#LoanAmount\r\n",
        "df = df.merge(pd.get_dummies(df.LoanAmount, drop_first= True, prefix='LoanAmount'), left_index=True, right_index=True)"
      ],
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xx5GYGs3XSlM"
      },
      "source": [
        "#Loan_Amount_Term\r\n",
        "df = df.merge(pd.get_dummies(df.Loan_Amount_Term, drop_first= True, prefix='Loan_Amount_Term'), left_index=True, right_index=True)"
      ],
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dM8cCvpYV8_4"
      },
      "source": [
        "Preview our created data frame"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 326
        },
        "id": "YMmGl2ALWAU4",
        "outputId": "56071dd9-75cc-43a2-befc-cea2eb13a567"
      },
      "source": [
        "df.head()"
      ],
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Gender</th>\n",
              "      <th>Married</th>\n",
              "      <th>Dependents</th>\n",
              "      <th>Education</th>\n",
              "      <th>Self_Employed</th>\n",
              "      <th>ApplicantIncome</th>\n",
              "      <th>CoapplicantIncome</th>\n",
              "      <th>LoanAmount</th>\n",
              "      <th>Loan_Amount_Term</th>\n",
              "      <th>Credit_History</th>\n",
              "      <th>Property_Area</th>\n",
              "      <th>Default_Status</th>\n",
              "      <th>sex_Male</th>\n",
              "      <th>sex_nonbinary</th>\n",
              "      <th>Married_Yes</th>\n",
              "      <th>Dependents_1</th>\n",
              "      <th>Dependents_2</th>\n",
              "      <th>Dependents_3+</th>\n",
              "      <th>Dependents_unclear</th>\n",
              "      <th>Education_Not Graduate</th>\n",
              "      <th>Self_Employed_Yes</th>\n",
              "      <th>Self_Employed_temporary</th>\n",
              "      <th>Credit_History_1.0</th>\n",
              "      <th>Credit_History_nohistory</th>\n",
              "      <th>Property_Area_Semiurban</th>\n",
              "      <th>Property_Area_Urban</th>\n",
              "      <th>ApplicantIncome_(2500.0, 3161.333]</th>\n",
              "      <th>ApplicantIncome_(3161.333, 3800.0]</th>\n",
              "      <th>ApplicantIncome_(3800.0, 4863.333]</th>\n",
              "      <th>ApplicantIncome_(4863.333, 6995.0]</th>\n",
              "      <th>ApplicantIncome_(6995.0, 81000.0]</th>\n",
              "      <th>CoapplicantIncome_(10000.0, 20000.0]</th>\n",
              "      <th>CoapplicantIncome_(20000.0, 30000.0]</th>\n",
              "      <th>CoapplicantIncome_(30000.0, 41667.0]</th>\n",
              "      <th>LoanAmount_(140, 280]</th>\n",
              "      <th>LoanAmount_(280, 320]</th>\n",
              "      <th>LoanAmount_(320, 460]</th>\n",
              "      <th>LoanAmount_(460, 700]</th>\n",
              "      <th>Loan_Amount_Term_(96, 192]</th>\n",
              "      <th>Loan_Amount_Term_(192, 288]</th>\n",
              "      <th>Loan_Amount_Term_(288, 384]</th>\n",
              "      <th>Loan_Amount_Term_(384, 480]</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Male</td>\n",
              "      <td>No</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>(4863.333, 6995.0]</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>(288, 384]</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>1</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>(3800.0, 4863.333]</td>\n",
              "      <td>(0.0, 10000.0]</td>\n",
              "      <td>(0.0, 140.0]</td>\n",
              "      <td>(288, 384]</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Rural</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>Yes</td>\n",
              "      <td>(2500.0, 3161.333]</td>\n",
              "      <td>NaN</td>\n",
              "      <td>(0.0, 140.0]</td>\n",
              "      <td>(288, 384]</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Male</td>\n",
              "      <td>Yes</td>\n",
              "      <td>0</td>\n",
              "      <td>Not Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>(2500.0, 3161.333]</td>\n",
              "      <td>(0.0, 10000.0]</td>\n",
              "      <td>(0.0, 140.0]</td>\n",
              "      <td>(288, 384]</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Male</td>\n",
              "      <td>No</td>\n",
              "      <td>0</td>\n",
              "      <td>Graduate</td>\n",
              "      <td>No</td>\n",
              "      <td>(4863.333, 6995.0]</td>\n",
              "      <td>NaN</td>\n",
              "      <td>(140.0, 280.0]</td>\n",
              "      <td>(288, 384]</td>\n",
              "      <td>1.0</td>\n",
              "      <td>Urban</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  Gender Married  ... Loan_Amount_Term_(288, 384] Loan_Amount_Term_(384, 480]\n",
              "0   Male      No  ...                           1                           0\n",
              "1   Male     Yes  ...                           1                           0\n",
              "2   Male     Yes  ...                           1                           0\n",
              "3   Male     Yes  ...                           1                           0\n",
              "4   Male      No  ...                           1                           0\n",
              "\n",
              "[5 rows x 42 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0iVtAkmxaHZD"
      },
      "source": [
        "#we exclude the other columns since we have new ones\r\n",
        "del df[\"Gender\"]\r\n",
        "del df[\"Married\"]\r\n",
        "del df[\"Dependents\"]\r\n",
        "del df[\"Education\"]\r\n",
        "del df[\"Self_Employed\"]\r\n",
        "del df[\"ApplicantIncome\"]\r\n",
        "del df[\"CoapplicantIncome\"]\r\n",
        "del df[\"LoanAmount\"]\r\n",
        "del df[\"Loan_Amount_Term\"]\r\n",
        "del df[\"Credit_History\"]\r\n",
        "del df[\"Property_Area\"]"
      ],
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 241
        },
        "id": "poeO50bTbWhi",
        "outputId": "e8d22248-d7f6-484a-f819-1360f5431a37"
      },
      "source": [
        "df.head()"
      ],
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Default_Status</th>\n",
              "      <th>sex_Male</th>\n",
              "      <th>sex_nonbinary</th>\n",
              "      <th>Married_Yes</th>\n",
              "      <th>Dependents_1</th>\n",
              "      <th>Dependents_2</th>\n",
              "      <th>Dependents_3+</th>\n",
              "      <th>Dependents_unclear</th>\n",
              "      <th>Education_Not Graduate</th>\n",
              "      <th>Self_Employed_Yes</th>\n",
              "      <th>Self_Employed_temporary</th>\n",
              "      <th>Credit_History_1.0</th>\n",
              "      <th>Credit_History_nohistory</th>\n",
              "      <th>Property_Area_Semiurban</th>\n",
              "      <th>Property_Area_Urban</th>\n",
              "      <th>ApplicantIncome_(2500.0, 3161.333]</th>\n",
              "      <th>ApplicantIncome_(3161.333, 3800.0]</th>\n",
              "      <th>ApplicantIncome_(3800.0, 4863.333]</th>\n",
              "      <th>ApplicantIncome_(4863.333, 6995.0]</th>\n",
              "      <th>ApplicantIncome_(6995.0, 81000.0]</th>\n",
              "      <th>CoapplicantIncome_(10000.0, 20000.0]</th>\n",
              "      <th>CoapplicantIncome_(20000.0, 30000.0]</th>\n",
              "      <th>CoapplicantIncome_(30000.0, 41667.0]</th>\n",
              "      <th>LoanAmount_(140, 280]</th>\n",
              "      <th>LoanAmount_(280, 320]</th>\n",
              "      <th>LoanAmount_(320, 460]</th>\n",
              "      <th>LoanAmount_(460, 700]</th>\n",
              "      <th>Loan_Amount_Term_(96, 192]</th>\n",
              "      <th>Loan_Amount_Term_(192, 288]</th>\n",
              "      <th>Loan_Amount_Term_(288, 384]</th>\n",
              "      <th>Loan_Amount_Term_(384, 480]</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   Default_Status  ...  Loan_Amount_Term_(384, 480]\n",
              "0               1  ...                            0\n",
              "1               0  ...                            0\n",
              "2               1  ...                            0\n",
              "3               1  ...                            0\n",
              "4               1  ...                            0\n",
              "\n",
              "[5 rows x 31 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UH1lL4Oobb7T"
      },
      "source": [
        "##Model preparation"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p0WTauX4beUZ",
        "outputId": "bddc7a8d-0b3a-45e8-f3b2-dd09b9c2a772"
      },
      "source": [
        "# dividing our dataset into features (X) and target (y)\r\n",
        "X = df.drop(columns = ['Default_Status']).values\r\n",
        "y = df['Default_Status'].values\r\n",
        "\r\n",
        "print(X.shape)\r\n",
        "print(y.shape)"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(611, 30)\n",
            "(611,)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tCT2Y0TFbyaW"
      },
      "source": [
        "# splitting our dataset into 80-20 train-test sets\r\n",
        "from sklearn.model_selection import train_test_split \r\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)"
      ],
      "execution_count": 28,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CO36_KMQcIku"
      },
      "source": [
        "Because we had earlier seen that we had an imbalanced dataset, we will create a balanced dataset by trying to resample our dataset using SMOTE (Synthetic minority Oversampling Technique). This technique works randomly picking a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "evqypQN0b7th",
        "outputId": "11db0410-cdc4-4637-f921-754567a44211"
      },
      "source": [
        "# creating a balanced dataset\r\n",
        "from imblearn.over_sampling import SMOTE\r\n",
        "smt = SMOTE()\r\n",
        "X_train, y_train = smt.fit_sample(X_train, y_train)"
      ],
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/externals/six.py:31: FutureWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).\n",
            "  \"(https://pypi.org/project/six/).\", FutureWarning)\n",
            "/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.neighbors.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.\n",
            "  warnings.warn(message, FutureWarning)\n",
            "/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function safe_indexing is deprecated; safe_indexing is deprecated in version 0.22 and will be removed in version 0.24.\n",
            "  warnings.warn(msg, category=FutureWarning)\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fsYu_8rucBSz",
        "outputId": "2ec54965-204e-42c8-c9a3-f37003cfe267"
      },
      "source": [
        "# we check the amount of records in each category\r\n",
        "np.bincount(y_train)"
      ],
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([335, 335])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QhO0OmnbcWAu"
      },
      "source": [
        "#MODELLING"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xF7OFaus54l1"
      },
      "source": [
        "###(a) Logistic Regression"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0m1AFloOcZDw",
        "outputId": "ad9fd1c5-d339-4761-c5d3-aa8af62ea5d0"
      },
      "source": [
        "# model creation\r\n",
        "from sklearn.linear_model import LogisticRegression\r\n",
        "logistic_classifier = LogisticRegression()\r\n",
        "\r\n",
        "# training our model\r\n",
        "logistic_classifier.fit(X_train, y_train)\r\n",
        "\r\n",
        "# making predictions\r\n",
        "y_pred_logistic = logistic_classifier.predict(X_test)"
      ],
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kXj5PDHxceQ2"
      },
      "source": [
        "#MODEL EVALUATION"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "G2IjOwLDch_8",
        "outputId": "c1b1cf96-7651-4c2d-c537-39529906895b"
      },
      "source": [
        "# model evaluation\r\n",
        "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\r\n",
        "print(accuracy_score(y_pred_logistic, y_test))\r\n",
        "print(confusion_matrix(y_test, y_pred_logistic))\r\n",
        "print(classification_report(y_test, y_pred_logistic))"
      ],
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0.7073170731707317\n",
            "[[ 3 36]\n",
            " [ 0 84]]\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      0.08      0.14        39\n",
            "           1       0.70      1.00      0.82        84\n",
            "\n",
            "    accuracy                           0.71       123\n",
            "   macro avg       0.85      0.54      0.48       123\n",
            "weighted avg       0.80      0.71      0.61       123\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_OHbHavrc2VF"
      },
      "source": [
        "The acccuracy of our model is 0.71\r\n",
        "\r\n",
        "From our confusion matrix, 3 records with class 0(not defaulting) were predicted correctly while 0 were predicted incorrectly. 84 of class 1(defaulting) were predicted correctly while 36 were predicted incorrectly.\r\n",
        "\r\n",
        "We have a recall of 0.54 (macro avg); ability to predict positive when it was actually positive.\r\n",
        "\r\n",
        "\r\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "DwB8WFQGcoOO",
        "outputId": "0b4f4914-ffbd-4230-ca7f-15e19e0ca7af"
      },
      "source": [
        "# Exploring another metric below \r\n",
        "# ---\r\n",
        "# plotting roc curve (receiving operating characteristic curve)\r\n",
        "from sklearn.metrics import roc_curve, roc_auc_score\r\n",
        "\r\n",
        "# Create true and false positive rates\r\n",
        "false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_pred_logistic)\r\n",
        "\r\n",
        "# Plot ROC curve\r\n",
        "plt.title('Receiver Operating Characteristic')\r\n",
        "plt.plot(false_positive_rate, true_positive_rate)\r\n",
        "plt.plot([0, 1], ls=\"--\")\r\n",
        "plt.plot([0, 0], [1, 0] , c=\".7\"), plt.plot([1, 1] , c=\".7\")\r\n",
        "plt.ylabel('True Positive Rate')\r\n",
        "plt.xlabel('False Positive Rate')\r\n",
        "plt.show()"
      ],
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zBJIi_0V4MCk"
      },
      "source": [
        "The above shows the true positive rate(recall) and false positive rate for every probability threshold of a binary classifier.\r\n",
        "The higher the blue line the better the model at distingushing beween the positive and negative classes."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MLc31oemdSIc",
        "outputId": "866f741a-d91f-4ab3-d79c-0b65e7ec6a76"
      },
      "source": [
        "# roc_auc_score\r\n",
        "roc_auc_score(y_test, y_pred_logistic)"
      ],
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.5384615384615384"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "O_D_-fzg5nFk"
      },
      "source": [
        "The model was fair."
      ]
    }
  ]
}