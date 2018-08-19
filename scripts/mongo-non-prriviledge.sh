# Detect the platform.
case "$OSTYPE" in

  solaris*)
    ;;
  
  linux-gnu*) # Debian
   curl -O https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-3.6.4.tgz
   tar -zxvf mongodb-linux-x86_64-3.6.4.tgz
   mkdir -p ~/.local/mongodb 
   cp -R -n mongodb-linux-x86_64-3.6.4/ ~/.local/mongodb
   export PATH=~/.local/mongodb/bin:$PATH
   source ~/.bashrc
   mkdir -p ~/.local/data/db
   mongod --dbpath ~/.local/data/db

  darwin*) # Mac (OSX)
   curl -O https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-3.6.4.tgz
   tar -zxvf mongodb-linux-x86_64-3.6.4.tgz
   mkdir -p ~/.local/mongodb 
   cp -R -n mongodb-linux-x86_64-3.6.4/ ~/.local/mongodb
   export PATH=~/.local/mongodb/bin:$PATH
   source ~/.bashrc
   mkdir -p ~/.local/data/db
   mongod --dbpath ~/.local/data/db
 
  *)
    echo "Unknown: $OSTYPE"
    ;;
esac
