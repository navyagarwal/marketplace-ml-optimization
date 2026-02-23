====================================================================================================
DATA EXPLORATION REPORT
====================================================================================================

Total CSV files found: 9


====================================================================================================
FILE: olist_customers_dataset.csv
====================================================================================================
Shape (rows, columns): (99441, 5)
File size: 8.62 MB

Columns and Data Types:
--------------------------------------------------------------------------------
  customer_id                              str                 
  customer_unique_id                       str                 
  customer_zip_code_prefix                 int64               
  customer_city                            str                 
  customer_state                           str                 

Missing Values:
--------------------------------------------------------------------------------
  No missing values found!

Basic Statistics:
--------------------------------------------------------------------------------
  Total rows: 99,441
  Total columns: 5
  Memory usage: 26.59 MB

First 5 rows:
--------------------------------------------------------------------------------
                        customer_id                customer_unique_id  \
0  06b8999e2fba1a1fbc88172c00ba8bc7  861eff4711a542e4b93843c6dd7febb0   
1  18955e83d337fd6b2def6b18a428ac77  290c77bc529b7ac935b93aa66c333dc3   
2  4e7b3e00288586ebd08712fdd0374a03  060e732b5b29e8181a18229c7b0b2b5e   
3  b2b6027bc5c5109e529d4dc6358b12c3  259dac757896d24d7702b9acbbff3f3c   
4  4f2d8ab171c80ec8364f7c12e35b23ad  345ecd01c38d18a9036ed96c73b8d066   

   customer_zip_code_prefix          customer_city customer_state  
0                     14409                 franca             SP  
1                      9790  sao bernardo do campo             SP  
2                      1151              sao paulo             SP  
3                      8775        mogi das cruzes             SP  
4                     13056               campinas             SP  

====================================================================================================
FILE: olist_geolocation_dataset.csv
====================================================================================================
Shape (rows, columns): (1000163, 5)
File size: 58.44 MB

Columns and Data Types:
--------------------------------------------------------------------------------
  geolocation_zip_code_prefix              int64               
  geolocation_lat                          float64             
  geolocation_lng                          float64             
  geolocation_city                         str                 
  geolocation_state                        str                 

Missing Values:
--------------------------------------------------------------------------------
  No missing values found!

Basic Statistics:
--------------------------------------------------------------------------------
  Total rows: 1,000,163
  Total columns: 5
  Memory usage: 129.38 MB

First 5 rows:
--------------------------------------------------------------------------------
   geolocation_zip_code_prefix  geolocation_lat  geolocation_lng  \
0                         1037       -23.545621       -46.639292   
1                         1046       -23.546081       -46.644820   
2                         1046       -23.546129       -46.642951   
3                         1041       -23.544392       -46.639499   
4                         1035       -23.541578       -46.641607   

  geolocation_city geolocation_state  
0        sao paulo                SP  
1        sao paulo                SP  
2        sao paulo                SP  
3        sao paulo                SP  
4        sao paulo                SP  

====================================================================================================
FILE: olist_order_items_dataset.csv
====================================================================================================
Shape (rows, columns): (112650, 7)
File size: 14.72 MB

Columns and Data Types:
--------------------------------------------------------------------------------
  order_id                                 str                 
  order_item_id                            int64               
  product_id                               str                 
  seller_id                                str                 
  shipping_limit_date                      str                 
  price                                    float64             
  freight_value                            float64             

Missing Values:
--------------------------------------------------------------------------------
  No missing values found!

Basic Statistics:
--------------------------------------------------------------------------------
  Total rows: 112,650
  Total columns: 7
  Memory usage: 35.99 MB

First 5 rows:
--------------------------------------------------------------------------------
                           order_id  order_item_id  \
0  00010242fe8c5a6d1ba2dd792cb16214              1   
1  00018f77f2f0320c557190d7a144bdd3              1   
2  000229ec398224ef6ca0657da4fc703e              1   
3  00024acbcdf0a6daa1e931b038114c75              1   
4  00042b26cf59d7ce69dfabb4e55b4fd9              1   

                         product_id                         seller_id  \
0  4244733e06e7ecb4970a6e2683c13e61  48436dade18ac8b2bce089ec2a041202   
1  e5f2d52b802189ee658865ca93d83a8f  dd7ddc04e1b6c2c614352b383efe2d36   
2  c777355d18b72b67abbeef9df44fd0fd  5b51032eddd242adc84c38acab88f23d   
3  7634da152a4610f1595efa32f14722fc  9d7a1d34a5052409006425275ba1c2b4   
4  ac6c3623068f30de03045865e4e10089  df560393f3a51e74553ab94004ba5c87   

   shipping_limit_date   price  freight_value  
0  2017-09-19 09:45:35   58.90          13.29  
1  2017-05-03 11:05:13  239.90          19.93  
2  2018-01-18 14:48:30  199.00          17.87  
3  2018-08-15 10:10:18   12.99          12.79  
4  2017-02-13 13:57:51  199.90          18.14  

====================================================================================================
FILE: olist_order_payments_dataset.csv
====================================================================================================
Shape (rows, columns): (103886, 5)
File size: 5.51 MB

Columns and Data Types:
--------------------------------------------------------------------------------
  order_id                                 str                 
  payment_sequential                       int64               
  payment_type                             str                 
  payment_installments                     int64               
  payment_value                            float64             

Missing Values:
--------------------------------------------------------------------------------
  No missing values found!

Basic Statistics:
--------------------------------------------------------------------------------
  Total rows: 103,886
  Total columns: 5
  Memory usage: 16.23 MB

First 5 rows:
--------------------------------------------------------------------------------
                           order_id  payment_sequential payment_type  \
0  b81ef226f3fe1789b1e8b2acac839d17                   1  credit_card   
1  a9810da82917af2d9aefd1278f1dcfa0                   1  credit_card   
2  25e8ea4e93396b6fa0d3dd708e76c1bd                   1  credit_card   
3  ba78997921bbcdc1373bb41e913ab953                   1  credit_card   
4  42fdf880ba16b47b59251dd489d4441a                   1  credit_card   

   payment_installments  payment_value  
0                     8          99.33  
1                     1          24.39  
2                     1          65.71  
3                     8         107.78  
4                     2         128.45  

====================================================================================================
FILE: olist_order_reviews_dataset.csv
====================================================================================================
Shape (rows, columns): (99224, 7)
File size: 13.78 MB

Columns and Data Types:
--------------------------------------------------------------------------------
  review_id                                str                 
  order_id                                 str                 
  review_score                             int64               
  review_comment_title                     str                 
  review_comment_message                   str                 
  review_creation_date                     str                 
  review_answer_timestamp                  str                 

Missing Values:
--------------------------------------------------------------------------------
  review_comment_title                          87656 ( 88.34%)
  review_comment_message                        58247 ( 58.70%)

Basic Statistics:
--------------------------------------------------------------------------------
  Total rows: 99,224
  Total columns: 7
  Memory usage: 39.12 MB

First 5 rows:
--------------------------------------------------------------------------------
                          review_id                          order_id  \
0  7bc2406110b926393aa56f80a40eba40  73fc7af87114b39712e6da79b0a377eb   
1  80e641a11e56f04c1ad469d5645fdfde  a548910a1c6147796b98fdf73dbeba33   
2  228ce5500dc1d8e020d8d1322874b6f0  f9e4b658b201a9f2ecdecbb34bed034b   
3  e64fb393e7b32834bb789ff8bb30750e  658677c97b385a9be170737859d3511b   
4  f7c4243c7fe1938f181bec41a392bdeb  8e6bfb81e283fa7e4f11123a3fb894f1   

   review_score review_comment_title  \
0             4                  NaN   
1             5                  NaN   
2             5                  NaN   
3             5                  NaN   
4             5                  NaN   

                              review_comment_message review_creation_date  \
0                                                NaN  2018-01-18 00:00:00   
1                                                NaN  2018-03-10 00:00:00   
2                                                NaN  2018-02-17 00:00:00   
3              Recebi bem antes do prazo estipulado.  2017-04-21 00:00:00   
4  Parabéns lojas lannister adorei comprar pela I...  2018-03-01 00:00:00   

  review_answer_timestamp  
0     2018-01-18 21:46:59  
1     2018-03-11 03:05:13  
2     2018-02-18 14:36:24  
3     2017-04-21 22:02:06  
4     2018-03-02 10:26:53  

====================================================================================================
FILE: olist_orders_dataset.csv
====================================================================================================
Shape (rows, columns): (99441, 8)
File size: 16.84 MB

Columns and Data Types:
--------------------------------------------------------------------------------
  order_id                                 str                 
  customer_id                              str                 
  order_status                             str                 
  order_purchase_timestamp                 str                 
  order_approved_at                        str                 
  order_delivered_carrier_date             str                 
  order_delivered_customer_date            str                 
  order_estimated_delivery_date            str                 

Missing Values:
--------------------------------------------------------------------------------
  order_approved_at                               160 (  0.16%)
  order_delivered_carrier_date                   1783 (  1.79%)
  order_delivered_customer_date                  2965 (  2.98%)

Basic Statistics:
--------------------------------------------------------------------------------
  Total rows: 99,441
  Total columns: 8
  Memory usage: 52.94 MB

First 5 rows:
--------------------------------------------------------------------------------
                           order_id                       customer_id  \
0  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d   
1  53cdb2fc8bc7dce0b6741e2150273451  b0830fb4747a6c6d20dea0b8c802d7ef   
2  47770eb9100c2d0c44946d9cf07ec65d  41ce2a54c0b03bf3443c3d931a367089   
3  949d5b44dbf5de918fe9c16f97b45f8a  f88197465ea7920adcdbec7375364d82   
4  ad21c59c0840e6cb83a9ceb5573f8159  8ab97904e6daea8866dbdbc4fb7aad2c   

  order_status order_purchase_timestamp    order_approved_at  \
0    delivered      2017-10-02 10:56:33  2017-10-02 11:07:15   
1    delivered      2018-07-24 20:41:37  2018-07-26 03:24:27   
2    delivered      2018-08-08 08:38:49  2018-08-08 08:55:23   
3    delivered      2017-11-18 19:28:06  2017-11-18 19:45:59   
4    delivered      2018-02-13 21:18:39  2018-02-13 22:20:29   

  order_delivered_carrier_date order_delivered_customer_date  \
0          2017-10-04 19:55:00           2017-10-10 21:25:13   
1          2018-07-26 14:31:00           2018-08-07 15:27:45   
2          2018-08-08 13:50:00           2018-08-17 18:06:29   
3          2017-11-22 13:39:59           2017-12-02 00:28:42   
4          2018-02-14 19:46:34           2018-02-16 18:17:02   

  order_estimated_delivery_date  
0           2017-10-18 00:00:00  
1           2018-08-13 00:00:00  
2           2018-09-04 00:00:00  
3           2017-12-15 00:00:00  
4           2018-02-26 00:00:00  

====================================================================================================
FILE: olist_products_dataset.csv
====================================================================================================
Shape (rows, columns): (32951, 9)
File size: 2.27 MB

Columns and Data Types:
--------------------------------------------------------------------------------
  product_id                               str                 
  product_category_name                    str                 
  product_name_lenght                      float64             
  product_description_lenght               float64             
  product_photos_qty                       float64             
  product_weight_g                         float64             
  product_length_cm                        float64             
  product_height_cm                        float64             
  product_width_cm                         float64             

Missing Values:
--------------------------------------------------------------------------------
  product_category_name                           610 (  1.85%)
  product_name_lenght                             610 (  1.85%)
  product_description_lenght                      610 (  1.85%)
  product_photos_qty                              610 (  1.85%)
  product_weight_g                                  2 (  0.01%)
  product_length_cm                                 2 (  0.01%)
  product_height_cm                                 2 (  0.01%)
  product_width_cm                                  2 (  0.01%)

Basic Statistics:
--------------------------------------------------------------------------------
  Total rows: 32,951
  Total columns: 9
  Memory usage: 6.30 MB

First 5 rows:
--------------------------------------------------------------------------------
                         product_id  product_category_name  \
0  1e9e8ef04dbcff4541ed26657ea517e5             perfumaria   
1  3aa071139cb16b67ca9e5dea641aaa2f                  artes   
2  96bd76ec8810374ed1b65e291975717f          esporte_lazer   
3  cef67bcfe19066a932b7673e239eb23d                  bebes   
4  9dc1a7de274444849c219cff195d0b71  utilidades_domesticas   

   product_name_lenght  product_description_lenght  product_photos_qty  \
0                 40.0                       287.0                 1.0   
1                 44.0                       276.0                 1.0   
2                 46.0                       250.0                 1.0   
3                 27.0                       261.0                 1.0   
4                 37.0                       402.0                 4.0   

   product_weight_g  product_length_cm  product_height_cm  product_width_cm  
0             225.0               16.0               10.0              14.0  
1            1000.0               30.0               18.0              20.0  
2             154.0               18.0                9.0              15.0  
3             371.0               26.0                4.0              26.0  
4             625.0               20.0               17.0              13.0  

====================================================================================================
FILE: olist_sellers_dataset.csv
====================================================================================================
Shape (rows, columns): (3095, 4)
File size: 0.17 MB

Columns and Data Types:
--------------------------------------------------------------------------------
  seller_id                                str                 
  seller_zip_code_prefix                   int64               
  seller_city                              str                 
  seller_state                             str                 

Missing Values:
--------------------------------------------------------------------------------
  No missing values found!

Basic Statistics:
--------------------------------------------------------------------------------
  Total rows: 3,095
  Total columns: 4
  Memory usage: 0.59 MB

First 5 rows:
--------------------------------------------------------------------------------
                          seller_id  seller_zip_code_prefix  \
0  3442f8959a84dea7ee197c632cb2df15                   13023   
1  d1b65fc7debc3361ea86b5f14c68d2e2                   13844   
2  ce3ad9de960102d0677a81f5d0bb7b2d                   20031   
3  c0f3eea2e14555b6faeea3dd58c1b1c3                    4195   
4  51a04a8a6bdcb23deccc82b0b80742cf                   12914   

         seller_city seller_state  
0           campinas           SP  
1         mogi guacu           SP  
2     rio de janeiro           RJ  
3          sao paulo           SP  
4  braganca paulista           SP  

====================================================================================================
FILE: product_category_name_translation.csv
====================================================================================================
Shape (rows, columns): (71, 2)
File size: 0.00 MB

Columns and Data Types:
--------------------------------------------------------------------------------
  product_category_name                    str                 
  product_category_name_english            str                 

Missing Values:
--------------------------------------------------------------------------------
  No missing values found!

Basic Statistics:
--------------------------------------------------------------------------------
  Total rows: 71
  Total columns: 2
  Memory usage: 0.01 MB

First 5 rows:
--------------------------------------------------------------------------------
    product_category_name product_category_name_english
0            beleza_saude                 health_beauty
1  informatica_acessorios         computers_accessories
2              automotivo                          auto
3         cama_mesa_banho                bed_bath_table
4        moveis_decoracao               furniture_decor

====================================================================================================
SUMMARY
====================================================================================================

Total datasets: 9
Total size: 120.34 MB
Total rows across all files: 1,550,922

Dataset Overview:
--------------------------------------------------------------------------------
  olist_customers_dataset.csv                       99,441 rows ×   5 cols |    8.62 MB
  olist_geolocation_dataset.csv                  1,000,163 rows ×   5 cols |   58.44 MB
  olist_order_items_dataset.csv                    112,650 rows ×   7 cols |   14.72 MB
  olist_order_payments_dataset.csv                 103,886 rows ×   5 cols |    5.51 MB
  olist_order_reviews_dataset.csv                   99,224 rows ×   7 cols |   13.78 MB
  olist_orders_dataset.csv                          99,441 rows ×   8 cols |   16.84 MB
  olist_products_dataset.csv                        32,951 rows ×   9 cols |    2.27 MB
  olist_sellers_dataset.csv                          3,095 rows ×   4 cols |    0.17 MB
  product_category_name_translation.csv                 71 rows ×   2 cols |    0.00 MB