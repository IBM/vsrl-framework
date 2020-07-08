zip -r assets.zip *;
scp assets.zip safelearning.ai:/var/www/html/vsrl/assets;
rm assets.zip
