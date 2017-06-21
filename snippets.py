from datetime import datetime, date, timedelta
datetime.now().date()
str(datetime.strptime('2016-06-21', '%Y-%m-%d').date())
lastyear = datetime.now() - timedelta(days=365)
