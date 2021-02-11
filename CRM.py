import datetime as dt
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import argparse
import time
import schedule


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def crm_data_prep(dataframe):
    dataframe.dropna(axis=0, inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    return dataframe


def create_rfm(dataframe):
    # CALCULATION OF RFM METRICS

    today_date = dt.datetime(2011, 12, 11)

    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})

    rfm.columns = ['recency', 'frequency', "monetary"]

    rfm = rfm[(rfm['monetary'] > 0)]

    # CALCULATION OF RFM SCORES
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    # NAMING SEGMENTS
    rfm['rfm_segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['rfm_segment'] = rfm['rfm_segment'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "rfm_segment"]]
    return rfm


def create_cltv_c(dataframe):
    # avg_order_value
    dataframe['avg_order_value'] = dataframe['monetary'] / dataframe['frequency']

    # purchase_frequency
    dataframe["purchase_frequency"] = dataframe['frequency'] / dataframe.shape[0]

    # repeat rate & churn rate
    repeat_rate = dataframe[dataframe.frequency > 1].shape[0] / dataframe.shape[0]
    churn_rate = 1 - repeat_rate

    # profit_margin
    dataframe['profit_margin'] = dataframe['monetary'] * 0.05

    # Customer Value
    dataframe['cv'] = (dataframe['avg_order_value'] * dataframe["purchase_frequency"])

    # Customer Lifetime Value
    dataframe['cltv'] = (dataframe['cv'] / churn_rate) * dataframe['profit_margin']

    # minmaxscaler
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(dataframe[["cltv"]])
    dataframe["cltv_c"] = scaler.transform(dataframe[["cltv"]])

    dataframe["cltv_c_segment"] = pd.qcut(dataframe["cltv_c"], 3, labels=["C", "B", "A"])

    dataframe = dataframe[["recency", "frequency", "monetary", "rfm_segment",
                           "cltv_c", "cltv_c_segment"]]

    return dataframe


def create_cltv_p(dataframe):
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (today_date - date.max()).days,
                                                                lambda date: (today_date - date.min()).days],
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = rfm.columns.droplevel(0)

    rfm.columns = ['recency', 'T', 'frequency', 'monetary']

    # CALCULATION OF MONETARY AVG & ADDING RFM INTO DF
    temp_df = dataframe.groupby(["Customer ID", "Invoice"]).agg({"TotalPrice": ["mean"]})
    temp_df = temp_df.reset_index()
    temp_df.columns = temp_df.columns.droplevel(0)
    temp_df.columns = ["Customer ID", "Invoice", "total_price_mean"]
    temp_df2 = temp_df.groupby(["Customer ID"], as_index=False).agg({"total_price_mean": ["mean"]})
    temp_df2.columns = temp_df2.columns.droplevel(0)
    temp_df2.columns = ["Customer ID", "monetary_avg"]

    rfm = rfm.merge(temp_df2, how="left", on="Customer ID")
    rfm.set_index("Customer ID", inplace=True)
    rfm.index = rfm.index.astype(int)

    # CALCULATION OF WEEKLY RECENCY AND WEEKLY T FOR BGNBD
    rfm["recency_weekly"] = rfm["recency"] / 7
    rfm["T_weekly"] = rfm["T"] / 7

    # CONTROL
    rfm = rfm[rfm["monetary_avg"] > 0]
    rfm["frequency"] = rfm["frequency"].astype(int)

    # BGNBD
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm['frequency'],
            rfm['recency_weekly'],
            rfm['T_weekly'])

    # exp_sales_1_month
    rfm["exp_sales_1_month"] = bgf.predict(4,
                                           rfm['frequency'],
                                           rfm['recency_weekly'],
                                           rfm['T_weekly'])
    # exp_sales_3_month
    rfm["exp_sales_3_month"] = bgf.predict(12,
                                           rfm['frequency'],
                                           rfm['recency_weekly'],
                                           rfm['T_weekly'])

    # expected_average_profit
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(rfm['frequency'], rfm['monetary_avg'])
    rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                             rfm['monetary_avg'])
    # 6 MONTHS cltv_p
    cltv = ggf.customer_lifetime_value(bgf,
                                       rfm['frequency'],
                                       rfm['recency_weekly'],
                                       rfm['T_weekly'],
                                       rfm['monetary_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)

    rfm["cltv_p"] = cltv

    # minmaxscaler
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(rfm[["cltv_p"]])
    rfm["cltv_p"] = scaler.transform(rfm[["cltv_p"]])

    # cltv_p_segment
    rfm["cltv_p_segment"] = pd.qcut(rfm["cltv_p"], 3, labels=["C", "B", "A"])

    rfm = rfm[["monetary_avg", "T", "recency_weekly", "T_weekly",
               "exp_sales_1_month", "exp_sales_3_month", "expected_average_profit",
               "cltv_p", "cltv_p_segment"]]

    return rfm


pd.set_option('display.max_columns', None)



ap = argparse.ArgumentParser()
ap.add_argument("--report", type=str,required=True, help="Do you want report?", default="yes")
args = ap.parse_args()


def main():
    if args.report in ["yes", "y", "report", "true"]:
        print("Reading dataset")
        start = time.time()
        df = pd.read_excel("/Users/cemistanbullu/Desktop/DSMLBC/datasets/online_retail_II.xlsx",
                           sheet_name="Year 2010-2011",
                           engine="openpyxl")
        #IF YOU WANT TO READ DATASET IN EXISTING DATABSE THEN YOU NEED TO USE THE FOLLOWING PART,
        # UNLESS YOU NEED TO READ DATASET IN YOUR LOCAL
        # creds = {'user': #####,
        #          'passwd': #####',
        #          'host': ######,
        #          'port': ######,
        #          'db': #######}
        #
        # # MySQL conection string.
        # connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
        #
        # # sqlalchemy engine for MySQL connection.
        # conn = create_engine(connstr.format(**creds))
        #
        # df_mysql = pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)

        end = time.time()
        print("Dataset is read. Elapsed time: " + str(end-start)+ " seconds")
        start1 = time.time()
        start = time.time()
        print("Proceses are started")
        print("Data prep is started ")
        df_prep = crm_data_prep(df)
        end = time.time()
        print("data prep is finished. Elapsed time: " + str(end - start)+ " seconds")
        start = time.time()
        print("Preparing RFM metrics")
        rfm = create_rfm(df_prep)
        end = time.time()
        print("RFM is finished. Elapsed time: " + str(end - start)+ " seconds")
        start = time.time()
        print("cltv calculation is started")
        rfm_cltv = create_cltv_c(rfm)
        end = time.time()
        print("CLTV calculation is finished. Elapsed time:" + str(end - start)+ " seconds")
        start2 = time.time()
        print("CLTV prediction is started ")
        rfm_cltv_p = create_cltv_p(df_prep)
        end2 = time.time()
        print("CLTV prediction is finished. Elapsed time: " + str(end2 - start2)+ " seconds")
        print("Merge is applied")
        crm_final = rfm_cltv.merge(rfm_cltv_p, on="Customer ID", how="left")
        crm_final.index.name = 'CustomerID'
        crm_final.index = crm_final.index.astype(int)

        # SENDING DATASET INTO DATASET IF YOU READ FROM DATABASE AT THE BEGINNING
        # crm_final.to_sql(name=#####,
        #                  con=####,
        #                  if_exists=#####,
        #                  index=####,
        #                  index_label=#####)

        end1 = time.time()
        print("All proceses are finished. Total elapsed time:" + str(end1 - start1) + " seconds")
        print(crm_final.head())
        return crm_final
    else:
        print("Report was not requested")


if __name__ == '__main__':
    main()
    schedule.every().hour.do(main)
    while 1:
        schedule.run_pending()
        time.sleep(1)

