# Detect the platform.
case "$OSTYPE" in

  solaris*)
    ;;
  
  linux-gnu*) # Debian
   sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2930ADAE8CAF5059EE73BB4B58712A2291FA4AD5
   echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.6 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.6.list
   sudo apt-get update
   sudo apt-get install -y mongodb-org
   sudo service mongod start
   
  darwin*) # Mac (OSX)
   # Try installing brew
   /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
   # Brew is already installed
   brew install mongodb
  
  *)
    echo "Unknown: $OSTYPE"
    ;;
esac
