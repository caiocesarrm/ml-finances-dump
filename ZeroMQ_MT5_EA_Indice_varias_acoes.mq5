//+------------------------------------------------------------------+
//|                                       ZeroMQ_MT4_EA_Template.mq4 |
//|                                    Copyright 2017, Darwinex Labs |
//|                                        https://www.darwinex.com/ |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, Darwinex Labs."
#property link      "https://www.darwinex.com/"
#property version   "1.00"
#property strict

// Required: MQL-ZMQ from https://github.com/dingmaotu/mql-zmq
#include <Zmq/Zmq.mqh>
#include <Trade\Trade.mqh>

extern string PROJECT_NAME = "DWX_ZeroMQ_Example";
extern string ZEROMQ_PROTOCOL = "tcp";
extern string HOSTNAME = "*";
extern int REP_PORT = 5555;
extern int PUSH_PORT = 5556;
extern int REP_PORT_SCRIPT = 5557;
extern int PUSH_PORT_SCRIPT = 5558;
extern int MILLISECOND_TIMER = 1;  // 1 millisecond

extern string t0 = "--- Trading Parameters ---";
extern int MagicNumber = 123456;
extern int MaximumOrders = 1;
extern double MaximumLotSize = 0.01;

CTrade trade;
datetime barraAntiga;
int contador_dados_enviados = 0;
bool novo_minuto = false;

// CREATE ZeroMQ Context
Context context(PROJECT_NAME);

// CREATE ZMQ_REP SOCKET
Socket repSocket(context,ZMQ_REP);
Socket repSocketScript(context,ZMQ_REP);

// CREATE ZMQ_PUSH SOCKET
Socket pushSocket(context,ZMQ_PUSH);
Socket pushSocketScript(context,ZMQ_PUSH);

// VARIABLES FOR LATER
uchar data_recebido[];
ZmqMsg request_zmq;
ZmqMsg request_zmq_script;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   //--- variables for function parameters
   int start = 0; // bar index
   int count = 1; // number of bars
   datetime time[]; // array storing the returned bar time
   //--- copy time 
   CopyTime(_Symbol,PERIOD_M1,start,count,time);
   //--- output result
   barraAntiga = time[0];
   
   EventSetMillisecondTimer(MILLISECOND_TIMER);     // Set Millisecond Timer to get client socket input
   
   Print("[REP] Binding MT4 Server to Socket on Port " + (string)REP_PORT + "..");   
   Print("[PUSH] Binding MT4 Server to Socket on Port " + (string)PUSH_PORT + "..");
   Print("[REP] Binding MT4 Server to Socket on Port " + (string)REP_PORT_SCRIPT + "..");   
   Print("[PUSH] Binding MT4 Server to Socket on Port " + (string)PUSH_PORT_SCRIPT + "..");
   
   repSocket.bind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, REP_PORT));
   pushSocket.bind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, PUSH_PORT));
   repSocketScript.bind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, REP_PORT_SCRIPT));
   pushSocketScript.bind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, PUSH_PORT_SCRIPT));
   
   /*
       Maximum amount of time in milliseconds that the thread will try to send messages 
       after its socket has been closed (the default value of -1 means to linger forever):
   */
   
   repSocket.setLinger(1000);  // 1000 milliseconds
   repSocketScript.setLinger(1000);
   
   /* 
      If we initiate socket.send() without having a corresponding socket draining the queue, 
      we'll eat up memory as the socket just keeps enqueueing messages.
      
      So how many messages do we want ZeroMQ to buffer in RAM before blocking the socket?
   */
   
   repSocket.setSendHighWaterMark(500);     // 5 messages only.
   repSocketScript.setSendHighWaterMark(500); 
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
//---
   Print("[REP] Unbinding MT4 Server from Socket on Port " + (string)REP_PORT + "..");
   repSocket.unbind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, REP_PORT));
   Print("[REP] Unbinding MT4 Server from Socket on Port " + (string)REP_PORT_SCRIPT + "..");
   repSocketScript.unbind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, REP_PORT_SCRIPT));
   
   Print("[PUSH] Unbinding MT4 Server from Socket on Port " + (string)PUSH_PORT + "..");
   pushSocket.unbind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, PUSH_PORT));
   Print("[PUSH] Unbinding MT4 Server from Socket on Port " + (string)PUSH_PORT + "..");
   pushSocketScript.unbind(StringFormat("%s://%s:%d", ZEROMQ_PROTOCOL, HOSTNAME, PUSH_PORT_SCRIPT));
   
}
//+------------------------------------------------------------------+
//| Expert timer function                                            |
//+------------------------------------------------------------------+
void OnTimer()
{
//---

   /*
      For this example, we need:
      1) socket.recv(request,true)
      2) MessageHandler() to process the request
      3) socket.send(reply)
   */
   
   // Get client's response, but don't wait.
   repSocket.recv(request_zmq,true);
   repSocketScript.recv(request_zmq_script,true);
   // MessageHandler() should go here.   
   ZmqMsg reply = MessageHandler(request_zmq);
   ZmqMsg replyScript = MessageHandlerScript(request_zmq_script);
   // socket.send(reply) should go here.
   repSocket.send(reply);
   repSocketScript.send(replyScript);
}
//+------------------------------------------------------------------+

ZmqMsg MessageHandler(ZmqMsg &request1) {
   
   // Output object
   ZmqMsg reply;
   
   // Message components for later.
   string components[];
   
   if(request1.size() > 0) {
      // Get data from request   
      ArrayResize(data_recebido, (int)request1.size());
      request1.getData(data_recebido);
      string dataStr = CharArrayToString(data_recebido);
      
      // Process data
      ParseZmqMessage(dataStr, components);
      
      // Interpret data
      InterpretZmqMessage(&pushSocket, components);
      
      // Construct response
      ZmqMsg ret(StringFormat("[SERVER] Processing: %s", dataStr));
      reply = ret;
      
   }
   else {
      // NO DATA RECEIVED
   }
   
   return(reply);
}


ZmqMsg MessageHandlerScript(ZmqMsg &request1) {
  
   
   // Output object
   ZmqMsg reply;
   
   // Message components for later.
   string components[];
   
   if(request1.size() > 0) {
   
      // Get data from request   
      ArrayResize(data_recebido, (int)request1.size());
      request1.getData(data_recebido);
      string dataStr = CharArrayToString(data_recebido);
      
      // Process data
      ParseZmqMessage(dataStr, components);
      
      // Interpret data
      InterpretZmqMessage(&pushSocketScript, components);
      
      // Construct response
      ZmqMsg ret(StringFormat("[SERVER] Processing: %s", dataStr));
      reply = ret;
      
   }
   else {
      // NO DATA RECEIVED
   }
   
   return(reply);
}

// Interpret Zmq Message and perform actions
void InterpretZmqMessage(Socket &pSocket, string& compArray[]) {

   Print("ZMQ: Interpreting Message..");
   
   // Message Structures:
   
   // 1) Trading
   // TRADE|ACTION|TYPE|SYMBOL|PRICE|SL|TP|COMMENT|TICKET
   // e.g. TRADE|OPEN|1|EURUSD|0|50|50|R-to-MetaTrader4|12345678
   
   // The 12345678 at the end is the ticket ID, for MODIFY and CLOSE.
   
   // 2) Data Requests
   
   // 2.1) RATES|SYMBOL   -> Returns Current Bid/Ask
   
   // 2.2) DATA|SYMBOL|TIMEFRAME|START_DATETIME|END_DATETIME
   
   // NOTE: datetime has format: D'2015.01.01 00:00'
   
   /*
      compArray[0] = TRADE or RATES
      If RATES -> compArray[1] = Symbol
      
      If TRADE ->
         compArray[0] = TRADE
         compArray[1] = ACTION (e.g. OPEN, MODIFY, CLOSE)
         compArray[2] = TYPE (e.g. OP_BUY, OP_SELL, etc - only used when ACTION=OPEN)
         
         // ORDER TYPES: 
         // https://docs.mql4.com/constants/tradingconstants/orderproperties
         
         // OP_BUY = 0
         // OP_SELL = 1
         // OP_BUYLIMIT = 2
         // OP_SELLLIMIT = 3
         // OP_BUYSTOP = 4
         // OP_SELLSTOP = 5
         
         compArray[3] = Symbol (e.g. EURUSD, etc.)
         compArray[4] = Open/Close Price (ignored if ACTION = MODIFY)
         compArray[5] = SL
         compArray[6] = TP
         compArray[7] = Trade Comment
   */
   
   int switch_action = 0;
   
   if(compArray[0] == "TRADE" && (compArray[1] == "OPEN" || compArray[1] == "ENCERRAR"))
      switch_action = 1;
   if(compArray[0] == "RATES")
      switch_action = 2;
   if(compArray[0] == "TRADE" && compArray[1] == "CLOSE")
      switch_action = 3;
   if(compArray[0] == "DATA")
      switch_action = 4;
   
   string ret = "";
   int ticket = -1;
   bool ans = false;
   MqlRates price_array[];
   ArraySetAsSeries(price_array, true);
   
   int price_count = 0;
   
   
   
   switch(switch_action) 
   {
      case 1: {
               string symbol_recebido = compArray[3];
               int posicoes_abertas = PositionsTotal();
               Print("Posicoes abertas = ", posicoes_abertas);
               Print("novo minuto = ", novo_minuto);
               if(posicoes_abertas > 0 && novo_minuto == true) {
                  if(!trade.PositionClose(symbol_recebido))
                  {
                     
                     //--- failure message
                     Print("PositionClose() method failed. Return code=",trade.ResultRetcode(),
                           ". Descrição do código: ",trade.ResultRetcodeDescription());
                  }
                  else
                  {
                     Print("PositionClose() method executed successfully. Return code=",trade.ResultRetcode(),
                           " (",trade.ResultRetcodeDescription(),")");
                  }
                  
                } 
                if(novo_minuto == true && compArray[1] == "OPEN") {
                  novo_minuto = false;
                  trade.SetExpertMagicNumber(MagicNumber);
                  InformPullClient(pSocket, "OPEN TRADE Instruction Received");
                  double volume=1;         // specify a trade operation volume
                  string symbol="WINJ19";    //specify the symbol, for which the operation is performed
                  int    digits=(int)SymbolInfoInteger(symbol,SYMBOL_DIGITS); // number of decimal places
                  double point=SymbolInfoDouble(symbol,SYMBOL_POINT);         // point
                  double bid=SymbolInfoDouble(symbol,SYMBOL_BID);             // current price for closing LONG
                  double SL=bid-50*point;                                   // unnormalized SL value
                  SL=NormalizeDouble(SL,digits);                              // normalizing Stop Loss
                  double TP=bid+100*point;                                   // unnormalized TP value
                  TP=NormalizeDouble(TP,digits);                              // normalizing Take Profit
                  //--- receive the current open price for LONG positions
                  double open_price=SymbolInfoDouble(symbol,SYMBOL_ASK);
                  string comment=StringFormat("Buy %s %G lots at %s, SL=%s TP=%s",
                                              symbol,volume,
                                              DoubleToString(open_price,digits),
                                              DoubleToString(SL,digits),
                                              DoubleToString(TP,digits));
                                              
                  if(!trade.Buy(1, symbol, open_price,SL, TP, comment))
                  {
                     //--- failure message
                     Print("Buy() method failed. Return code=",trade.ResultRetcode(),
                           ". Descrição do código: ",trade.ResultRetcodeDescription());
                  }
                  else
                  {
                     Print("Buy() method executed successfully. Return code=",trade.ResultRetcode(),
                           " (",trade.ResultRetcodeDescription(),")");
                   }
               }
         }
         break;
      case 2: 
        
         break;
      case 3:
         InformPullClient(pSocket, "CLOSE TRADE Instruction Received");
         
         // IMPLEMENT CLOSE TRADE LOGIC HERE
         
         ret = StringFormat("Trade Closed (Ticket: %d)", ticket);
         InformPullClient(pSocket, ret);
         
         break;
      
      case 4: {
         InformPullClient(pSocket, "HISTORICAL DATA Instruction Received");
            //--- variables for function parameters
            int start = 0; // bar index
            int count = 1; // number of bars
            datetime time1[]; // array storing the returned bar time
            //--- copy time 
            CopyTime("WINJ19",PERIOD_M1,start,count,time1);
         if(barraAntiga != time1[0]) {
            contador_dados_enviados = 0;
         }
         if(barraAntiga != time1[0] || contador_dados_enviados < 8) {
            barraAntiga = time1[0];
            novo_minuto = true;
            Print(novo_minuto);
            contador_dados_enviados++;
            // Format: DATA|SYMBOL|TIMEFRAME|START_DATETIME|END_DATETIME
            //price_count = CopyClose(compArray[1], StringToInteger(compArray[2]), 
            //               StringToTime(compArray[3]), StringToTime(compArray[4]), 
            //               price_array);
            string symbol_recebido = compArray[1];
            
            //int periodo_recebido = (int)StringToInteger(compArray[2]);
            //receber apenas o close, usa CopyClose
            //Para receber o OHLC usar CopyRates
            price_count = CopyRates(symbol_recebido, PERIOD_M1, 0, 1,  price_array);
            
            if (price_count > 0) {
               
               ret = "";
               
               // Construct string of price|price|price|.. etc and send to PULL client.
               for(int i = 0; i < price_count; i++ ) {
     
                  if(i == 0)
                     ret = TimeToString(price_array[i].time) + "," + DoubleToString(price_array[i].close, 5);
                  else if(i > 0) {
                     ret = ret + "|" + DoubleToString(price_array[i].close, 5);
                     printf(ret);
                  }   
               }
               Print("Sending: " + ret);
               
               InformReplyClientScript(&repSocketScript, StringFormat("%s", ret));

               // Send data to PULL client.
               //InformPullClient(pSocket, StringFormat("%s", ret));
               // ret = "";
            }
         }
         else {
            string vazia = "";
            InformReplyClientScript(&repSocketScript, StringFormat("%s", vazia));
         }
      }
         break;
         
      default: 
         break;
   }
}

// Parse Zmq Message
void ParseZmqMessage(string& message, string& retArray[]) {
   
   Print("Parsing: " + message);
   
   string sep = "|";
   ushort u_sep = StringGetCharacter(sep,0);
   
   int splits = StringSplit(message, u_sep, retArray);
   
   for(int i = 0; i < splits; i++) {
      Print((string)i + ") " + (string)retArray[i]);
   }
}

//+------------------------------------------------------------------+

// Inform Client
void InformPullClient(Socket& pushSocket1, string message) {

   ZmqMsg pushReply(StringFormat("%s", message));
   // pushSocket.send(pushReply,true,false);
   
   pushSocket1.send(pushReply,true); // NON-BLOCKING
   // pushSocket.send(pushReply,false); // BLOCKING
   
}

void InformReplyClient(Socket& replySocket1, string message) {
   
   
   ZmqMsg ret(StringFormat("%s", message));
   ZmqMsg reply = ret;
   
   // socket.send(reply) should go here.
   replySocket1.send(reply);
}

void InformReplyClientScript(Socket& replySocket1, string message) {
   
   
   ZmqMsg ret(StringFormat("%s", message));
   ZmqMsg reply = ret;
   
   // socket.send(reply) should go here.
   replySocket1.send(reply);
}